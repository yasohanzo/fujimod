//! Borrowed diagnostic callbacks: swap one callback word, query, restore once.
use crate::{
    Error, Result,
    ff80::{
        commands::{
            diagnostic::{CommandStatus, DiagnosticReply, DiagnosticTag},
            memory::{Address, DebugMemoryLimits, MemoryRange},
        },
        sequence::SequenceCommand,
    },
};
use std::num::NonZeroU32;

/// Commands before the first call: the slot and output guards.
const GUARDS: usize = 2;
/// Commands per call: output, slot, query, slot guard, captured output.
const PER_CALL: usize = 5;
/// Commands after the last call: restore both words, then verify them.
const RESTORE: usize = 4;

/// A checked borrow of one callback word and its output buffer. Callback
/// authorization and code/cache qualification remain the caller's policy.
///
/// The whole batch is one checked sequence: the USB owner guards the stock
/// slot and original output, runs every call, restores both words and verifies
/// their final values, all with a single host wait. Queries accept any
/// completed reply at the USB owner so restoration always follows a returning
/// callback; the host checks each call's status after the batch returns. A
/// rejected status therefore does not stop later calls in the same batch: batch
/// only calls that are safe to run whatever an earlier call returned. After
/// every query the USB owner checks that the slot still holds the borrowed
/// callback (or the exact stock value for an explicitly self-restoring call);
/// another writer's value stops the batch and is never overwritten.
#[derive(Clone)]
pub struct BorrowedQueries {
    callback: Address,
    stock: Address,
    output: MemoryRange,
    original: Vec<u8>,
    calls: Vec<(Address, DiagnosticTag, Vec<(u32, u32)>, Vec<u8>, bool)>,
}
impl BorrowedQueries {
    pub fn new(
        callback: Address,
        stock: Address,
        output: Address,
        original: Vec<u8>,
    ) -> Result<Self> {
        let length = u32::try_from(original.len())
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(Error::InvalidLength)?;
        let output = MemoryRange::new(output, length)?;
        MemoryRange::new(callback, NonZeroU32::new(4).expect("word"))?;
        if u64::from(callback.0) < u64::from(output.address().0) + u64::from(length.get())
            && u64::from(output.address().0) < u64::from(callback.0) + 4
        {
            return Err(Error::OutsideDataWindow);
        }
        Ok(Self {
            callback,
            stock,
            output,
            original,
            calls: vec![],
        })
    }
    fn read(address: Address, length: u32, expected: Option<Vec<u8>>) -> SequenceCommand {
        SequenceCommand::Read {
            range: MemoryRange::new(
                address,
                NonZeroU32::new(length).expect("validated nonempty range"),
            )
            .expect("validated range"),
            expected,
        }
    }
    /// Append one callback; return indices of its raw query reply and output.
    /// Restoration is appended once, by `commands()`, after the entire batch.
    pub fn call(
        &mut self,
        callback: Address,
        tag: DiagnosticTag,
        statuses: Vec<(u32, u32)>,
        output: Vec<u8>,
    ) -> Result<(usize, usize)> {
        if output.len() != self.original.len() || statuses.is_empty() || statuses.len() > 8 {
            return Err(Error::InvalidLength);
        }
        let requested = GUARDS + PER_CALL * (self.calls.len() + 1) + RESTORE;
        if requested > 512 {
            return Err(Error::LengthLimit {
                requested: requested as u32,
                limit: 512,
            });
        }
        let first = GUARDS + PER_CALL * self.calls.len();
        self.calls.push((callback, tag, statuses, output, false));
        Ok((first + 2, first + 4))
    }
    /// Append a qualified callback that restores its own slot before returning.
    /// The caller must independently qualify that firmware ordering. After the
    /// query, the sequence requires the exact stock value; an unexpected value
    /// stops the batch before any restoration writes. This is not permission
    /// to treat a disconnected query as complete or to retry it.
    pub fn call_restoring_slot(
        &mut self,
        callback: Address,
        tag: DiagnosticTag,
        statuses: Vec<(u32, u32)>,
        output: Vec<u8>,
    ) -> Result<(usize, usize)> {
        let indices = self.call(callback, tag, statuses, output)?;
        self.calls.last_mut().expect("appended call").4 = true;
        Ok(indices)
    }

    /// Run the batch as one checked sequence. Return the raw query reply and
    /// captured output for each call. A call whose device status is not one it
    /// accepts fails after the batch has restored and verified both words.
    pub async fn invoke(
        &self,
        session: &mut crate::ff80::Session,
        limits: DebugMemoryLimits,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let commands = self.commands();
        if commands.is_empty() {
            return Ok(Vec::new());
        }
        let replies = session.sequence(commands, limits).await?;
        let mut results = Vec::with_capacity(self.calls.len());
        for (index, (_, _, statuses, _, _)) in self.calls.iter().enumerate() {
            let first = GUARDS + PER_CALL * index;
            let reply = replies[first + 2].clone();
            let status = match DiagnosticReply::decode(reply.clone(), 0x450006)?.status {
                CommandStatus::Success => (0, 0),
                CommandStatus::DeviceError { primary, secondary } => (primary, secondary),
            };
            if !statuses.contains(&status) {
                return Err(crate::error::ProtocolError::ExpectationMismatch.into());
            }
            results.push((reply, replies[first + 4].clone()));
        }
        Ok(results)
    }
    pub fn commands(&self) -> Vec<SequenceCommand> {
        if self.calls.is_empty() {
            return Vec::new();
        }
        let stock = self.stock.0.to_le_bytes().to_vec();
        let length = self.output.length().get();
        let mut commands = vec![
            Self::read(self.callback, 4, Some(stock.clone())),
            Self::read(self.output.address(), length, Some(self.original.clone())),
        ];
        for (callback, tag, _, output, restores_slot) in &self.calls {
            commands.extend([
                SequenceCommand::Write {
                    address: self.output.address(),
                    bytes: output.clone(),
                },
                SequenceCommand::Write {
                    address: self.callback,
                    bytes: callback.0.to_le_bytes().to_vec(),
                },
                SequenceCommand::Query {
                    tag: *tag,
                    statuses: None,
                },
                Self::read(
                    self.callback,
                    4,
                    Some(if *restores_slot {
                        stock.clone()
                    } else {
                        callback.0.to_le_bytes().to_vec()
                    }),
                ),
                Self::read(self.output.address(), length, None),
            ]);
        }
        commands.extend([
            SequenceCommand::Write {
                address: self.callback,
                bytes: stock.clone(),
            },
            SequenceCommand::Write {
                address: self.output.address(),
                bytes: self.original.clone(),
            },
            Self::read(self.callback, 4, Some(stock)),
            Self::read(self.output.address(), length, Some(self.original.clone())),
        ]);
        commands
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ff80::{
            SessionState,
            commands::memory::{ReadMemory, WriteMemory},
            framing::{RequestKind, envelope},
        },
        testing::{Exchange as E, Script},
    };
    use futures_lite::future::block_on;

    fn limits() -> DebugMemoryLimits {
        DebugMemoryLimits {
            read: NonZeroU32::new(64).unwrap(),
            write: NonZeroU32::new(64).unwrap(),
        }
    }
    fn command(steps: &mut Vec<E>, request: Vec<u8>, reply: Vec<u8>) {
        steps.extend([
            E::Control(envelope(
                RequestKind::WriteCommand,
                request.len() as u32,
                42,
            )),
            E::Write(request),
            E::Control(envelope(RequestKind::ReadCommand, reply.len() as u32, 42)),
            E::Read {
                maximum: reply.len(),
                data: reply,
            },
        ]);
    }
    fn read(steps: &mut Vec<E>, address: u32, bytes: &[u8]) {
        let range = MemoryRange::new(
            Address(address),
            NonZeroU32::new(bytes.len() as u32).unwrap(),
        )
        .unwrap();
        let request = ReadMemory::new(range, limits()).unwrap().encode();
        let mut reply = request.clone();
        reply.extend_from_slice(bytes);
        command(steps, request, reply);
    }
    fn write(steps: &mut Vec<E>, address: u32, bytes: &[u8]) {
        let request = WriteMemory::new(Address(address), bytes.to_vec(), limits())
            .unwrap()
            .encode();
        command(steps, request.clone(), request[..36].to_vec());
    }
    fn query_reply(tag: u32, status: (u32, u32)) -> Vec<u8> {
        let mut reply = crate::ff80::commands::diagnostic::query(DiagnosticTag::Research(tag));
        reply[4..8].copy_from_slice(&status.0.to_le_bytes());
        reply[8..12].copy_from_slice(&status.1.to_le_bytes());
        reply
    }
    /// The complete exchange for one call that replies with `status`.
    fn script(status: (u32, u32)) -> Script {
        script_with_slot(status, 0x8000)
    }
    fn script_with_slot(status: (u32, u32), after_slot: u32) -> Script {
        let mut s = vec![
            E::Control(envelope(RequestKind::Open, 8, 0)),
            E::Read {
                maximum: 8,
                data: vec![42, 0, 0, 0, 0, 0, 0, 64],
            },
        ];
        read(&mut s, 0x1000, &0x9000u32.to_le_bytes());
        read(&mut s, 0x1100, b"original");
        write(&mut s, 0x1100, &[0; 8]);
        write(&mut s, 0x1000, &0x8000u32.to_le_bytes());
        let request = crate::ff80::commands::diagnostic::query(DiagnosticTag::Research(7));
        command(&mut s, request, query_reply(7, status));
        read(&mut s, 0x1000, &after_slot.to_le_bytes());
        read(&mut s, 0x1100, b"captured");
        write(&mut s, 0x1000, &0x9000u32.to_le_bytes());
        write(&mut s, 0x1100, b"original");
        read(&mut s, 0x1000, &0x9000u32.to_le_bytes());
        read(&mut s, 0x1100, b"original");
        Script::new(s)
    }
    fn plan() -> BorrowedQueries {
        let mut plan = BorrowedQueries::new(
            Address(0x1000),
            Address(0x9000),
            Address(0x1100),
            b"original".to_vec(),
        )
        .unwrap();
        plan.call(
            Address(0x8000),
            DiagnosticTag::Research(7),
            vec![(0, 0)],
            vec![0; 8],
        )
        .unwrap();
        plan
    }

    #[test]
    fn borrowed_queries_validate_before_append_and_restore_once() {
        let mut plan = BorrowedQueries::new(
            Address(0x1000),
            Address(0x9000),
            Address(0x1100),
            b"original".to_vec(),
        )
        .unwrap();
        assert!(
            plan.call(
                Address(0x8000),
                DiagnosticTag::Research(7),
                vec![],
                vec![0; 8]
            )
            .is_err()
        );
        assert_eq!(plan.commands().len(), 0);
        assert_eq!(
            plan.call(
                Address(0x8000),
                DiagnosticTag::Research(7),
                vec![(0, 0), (1, 0xffff)],
                vec![0; 8]
            )
            .unwrap(),
            (4, 6)
        );
        assert_eq!(
            plan.call(
                Address(0x8004),
                DiagnosticTag::Research(9),
                vec![(0, 0)],
                vec![0; 8]
            )
            .unwrap(),
            (9, 11)
        );
        let commands = plan.commands();
        assert_eq!(commands.len(), 16);
        assert!(matches!(
            commands[4],
            SequenceCommand::Query { statuses: None, .. }
        ));
        let SequenceCommand::Write { address, bytes } = &commands[12] else {
            panic!()
        };
        assert_eq!(address.0, 0x1000);
        assert_eq!(bytes, &0x9000u32.to_le_bytes());
        let SequenceCommand::Read { expected, .. } = &commands[15] else {
            panic!()
        };
        assert_eq!(expected.as_deref(), Some(&b"original"[..]));
        assert!(
            BorrowedQueries::new(Address(0x1000), Address(0), Address(0x1002), vec![0; 8]).is_err()
        );
    }

    #[test]
    fn query_batches_refuse_overflow_without_mutating_the_plan() {
        let mut plan =
            BorrowedQueries::new(Address(0x1000), Address(0), Address(0x1100), vec![0; 8]).unwrap();
        for _ in 0..101 {
            plan.call(
                Address(0x8000),
                DiagnosticTag::Research(0),
                vec![(0, 0)],
                vec![0; 8],
            )
            .unwrap();
        }
        assert_eq!(plan.commands().len(), 511);
        assert!(
            plan.call(
                Address(0x8000),
                DiagnosticTag::Research(0),
                vec![(0, 0)],
                vec![0; 8]
            )
            .is_err()
        );
        assert_eq!(plan.commands().len(), 511);
    }

    #[test]
    fn one_sequence_guards_calls_restores_and_verifies() {
        let script = script((0, 0));
        let mut session = block_on(script.begin_session(256, None)).unwrap();
        let before = session.round_trips();
        let results = block_on(plan().invoke(&mut session, limits())).unwrap();
        assert_eq!(session.round_trips() - before, 1);
        assert_eq!(
            results,
            vec![(query_reply(7, (0, 0)), b"captured".to_vec())]
        );
        assert_eq!(script.remaining(), 0);
    }

    #[test]
    fn unexpected_status_fails_after_restoration_and_keeps_the_session() {
        let script = script((1, 0xffff));
        let mut session = block_on(script.begin_session(256, None)).unwrap();
        assert!(block_on(plan().invoke(&mut session, limits())).is_err());
        assert_eq!(script.remaining(), 0);
        assert_eq!(session.state(), SessionState::Ready);
    }
    fn restoring_plan() -> BorrowedQueries {
        let mut p = BorrowedQueries::new(
            Address(0x1000),
            Address(0x9000),
            Address(0x1100),
            b"original".to_vec(),
        )
        .unwrap();
        p.call_restoring_slot(
            Address(0x8000),
            DiagnosticTag::Research(7),
            vec![(0, 0)],
            vec![0; 8],
        )
        .unwrap();
        p
    }

    #[test]
    fn self_restored_slot_is_checked_before_output_cleanup() {
        let script = script_with_slot((0, 0), 0x9000);
        let mut session = block_on(script.begin_session(256, None)).unwrap();
        let results = block_on(restoring_plan().invoke(&mut session, limits())).unwrap();
        assert_eq!(
            results,
            vec![(query_reply(7, (0, 0)), b"captured".to_vec())]
        );
        assert_eq!(script.remaining(), 0);
        assert_eq!(session.state(), SessionState::Ready);
    }

    #[test]
    fn self_restoring_call_refuses_borrowed_or_foreign_slot_before_writes() {
        for after_slot in [0x8000, 0xdeadbeef] {
            let script = script_with_slot((0, 0), after_slot);
            let mut session = block_on(script.begin_session(256, None)).unwrap();
            assert!(block_on(restoring_plan().invoke(&mut session, limits())).is_err());
            // Five commands remain: capture output, restore slot/output, and
            // verify both. Each command consumes four transport exchanges.
            assert_eq!(script.remaining(), 20);
            assert_eq!(session.state(), SessionState::CompletionUnknown);
        }
    }
}
