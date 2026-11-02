use super::*;

#[derive(Deserialize)]
struct CapturedPaint {
    fill: FillDraw,
    address: u64,
    width: u32,
    height: u32,
    coverage_index: usize,
    samples: u32,
    clips: Vec<[u32; 4]>,
    order: u32,
    tiled: bool,
    flip: bool,
    blend: u32,
    texture: Option<Texture>,
    gradient: Option<Gradient>,
    solid: [u8; 4],
}

#[test]
#[ignore = "requires a local SMAPH failure capture and GPU adapter"]
fn captured_resident_batch_matches_cpu_per_draw() {
    type Targets = BTreeMap<u64, (u32, u32, Vec<u8>)>;
    type Capture = (
        u32,
        u64,
        usize,
        serde::de::IgnoredAny,
        serde::de::IgnoredAny,
        Vec<CapturedPaint>,
        Vec<[usize; 4]>,
        Vec<Vec<u32>>,
        Targets,
        Option<Targets>,
        Vec<u8>,
    );
    let path = std::env::var_os("FUJIMOD_SMAPH_REPLAY_CAPTURE").expect("capture path required");
    let reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let (version, _, _, _, _, paints, bounds, coverages, original, _, _): Capture =
        ciborium::from_reader(reader).unwrap();
    assert_eq!(version, 1);
    let context = fujimod_emulator_gpu::context().unwrap();
    let _guard = context.acquire().unwrap();
    let mut cpu = original.clone();
    let mut surfaces = BTreeMap::new();
    for (index, p) in paints.into_iter().enumerate() {
        let key = fujimod_emulator_gpu::SurfaceKey {
            guest_address: p.address,
            format: fujimod_emulator_gpu::SurfaceFormat::Rgba8,
            extent: fujimod_emulator_gpu::SurfaceExtent {
                width: p.width,
                height: p.height,
            },
            layout: if p.tiled {
                fujimod_emulator_gpu::SurfaceLayout::Morton8x8
            } else {
                fujimod_emulator_gpu::SurfaceLayout::Linear
            },
        };
        let surface = surfaces.entry(p.address).or_insert_with(|| {
            context
                .get_or_create_target(key, Some(&original[&p.address].2))
                .unwrap()
        });
        let b = bounds[p.coverage_index];
        let coverage = &coverages[p.coverage_index];
        let input = scanline::CoverageInput::new(&[], b, p.samples, None, false);
        let draw = PreparedPaint {
            fill: &p.fill,
            address: p.address,
            width: p.width,
            height: p.height,
            coverage_index: 0,
            samples: p.samples,
            clips: p.clips,
            order: p.order,
            tiled: p.tiled,
            flip: p.flip,
            blend: p.blend,
            texture: p.texture,
            gradient: p.gradient,
            solid: p.solid,
        };
        paint_cpu_draws(
            std::slice::from_ref(&draw),
            &[input],
            std::slice::from_ref(coverage),
            &mut cpu,
        )
        .unwrap();
        gpu_paint::paint_resident(
            surface,
            coverage,
            b,
            [p.width as usize, p.height as usize],
            &original[&p.address].2,
            p.samples,
            &p.fill.stream.regs,
            &p.fill.stream.gamma,
            p.order,
            p.tiled,
            p.flip,
            &draw.clips,
            p.solid,
            p.blend,
            draw.texture.as_ref(),
            draw.gradient.as_ref(),
            &p.fill.stream.gradient_deltas,
        )
        .unwrap();
        let actual = context.readback(key).unwrap().bytes;
        let expected = &cpu[&p.address].2;
        if let Some(offset) = expected.iter().zip(&actual).position(|(a, b)| a != b) {
            panic!(
                "draw {index}, target {:#x}, byte {offset}: CPU {}, GPU {}; bounds {b:?}",
                p.address, expected[offset], actual[offset]
            );
        }
        println!("draw {index} matched");
    }
}

#[test]
#[ignore = "requires a local coverage capture and GPU adapter"]
fn captured_coverage_batch_matches_cpu() {
    type Record = ([usize; 4], u32, Vec<[i32; 4]>, Option<i32>, bool);
    let path =
        std::env::var_os("FUJIMOD_SMAPH_REPLAY_COVERAGE").expect("coverage capture required");
    let reader = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    let (version, records): (u32, Vec<Record>) = ciborium::from_reader(reader).unwrap();
    assert_eq!(version, 1);
    let inputs: Vec<_> = records
        .into_iter()
        .map(
            |(bounds, samples, edges, stroke_radius, nonzero)| scanline::CoverageInput {
                bounds,
                samples,
                edges,
                stroke_radius,
                nonzero,
            },
        )
        .collect();
    let context = fujimod_emulator_gpu::context().unwrap();
    let _guard = context.acquire().unwrap();
    let started = std::time::Instant::now();
    let gpu = gpu_coverage::coverage_batch(&inputs).unwrap();
    println!(
        "GPU coverage for {} jobs: {:?}",
        inputs.len(),
        started.elapsed()
    );
    for (index, (input, actual)) in inputs.iter().zip(gpu).enumerate() {
        let expected = scanline::coverage_image(input);
        assert_eq!(actual.len(), expected.len(), "job {index}");
        if let Some(offset) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
            panic!(
                "coverage job {index}, sample {offset}: GPU {}, CPU {}",
                actual[offset], expected[offset]
            );
        }
        println!("coverage job {index} matched");
    }
}
