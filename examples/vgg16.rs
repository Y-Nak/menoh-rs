extern crate image;
extern crate menoh;

use image::GenericImage;
use std::io::BufRead;
use std::path::Path;

const CONV1_1_IN_NAME: &str = "140326425860192";
const INPUT_BATCH_SIZE: i32 = 1;
const INPUT_CHANNEL_NUM: i32 = 3;
const INPUT_WIDTH: i32 = 224;
const INPUT_HEIGHT: i32 = 224;
const SOFTMAX_OUT_NAME: &str = "140326200803680";

const MODEL_PATH: &str = "model/VGG16.onnx";
const CATEGORY_LIST_PATH: &str = "model/synset_words.txt";

const HEN_IMAGE_PATH: &str = "data/Light_sussex_hen.jpg";
const CAT_IMAGE_PATH: &str = "data/Feral_cat_Virginia_crop.jpg";

fn resize_im(mut im: image::DynamicImage, width: u32, height: u32) -> image::DynamicImage {
    let im_w = im.width();
    let im_h = im.height();
    let shortest_edge = std::cmp::min(im_h, im_w);
    let im = im.crop(
        (im_w - shortest_edge) / 2,
        (im_h - shortest_edge) / 2,
        shortest_edge,
        shortest_edge,
    );
    im.resize_exact(width, height, image::FilterType::Nearest)
}

fn reorder_to_chw(im: &image::DynamicImage) -> Vec<f32> {
    let (im_h, im_w) = (im.height(), im.width());

    let mut input_im: Vec<f32> = vec![Default::default(); (im_h * im_w * 3) as usize];
    for h in 0..im_h {
        for w in 0..im_w {
            for c in 0..3 {
                input_im[(c * (im_h * im_w) + h * im_w + w) as usize] =
                    im.get_pixel(h as u32, w as u32)[2 - c as usize] as f32;
            }
        }
    }
    input_im
}

fn to_input_vec(im: image::DynamicImage) -> Vec<f32> {
    let color = im.color();
    assert_eq!(color, image::RGB(8));
    let im = resize_im(im, INPUT_WIDTH as u32, INPUT_HEIGHT as u32);

    reorder_to_chw(&im)
}

fn print_top_category(result: &[f32], categories: &[String]) {
    let result = result
        .iter()
        .zip(categories)
        .max_by(|&(x, _), &(y, _)| x.partial_cmp(y).unwrap())
        .unwrap();

    println!("Prob: {}, Category: {} ", result.0, result.1)
}

fn parse_category<P>(p: P) -> Vec<String>
where
    P: AsRef<Path>,
{
    let file = std::fs::File::open(p).unwrap();
    let buf = std::io::BufReader::new(file);
    buf.lines().map(|l| l.unwrap()).collect()
}

fn main() {
    // load category file
    let categories = parse_category(CATEGORY_LIST_PATH);

    let model_data = menoh::ModelData::new(MODEL_PATH).unwrap();
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();

    let input_dims = vec![
        INPUT_BATCH_SIZE,
        INPUT_CHANNEL_NUM,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    ];

    vpt_builder
        .add_input_profile(CONV1_1_IN_NAME, menoh::Dtype::Float, &input_dims)
        .unwrap();

    vpt_builder
        .add_output_profile(SOFTMAX_OUT_NAME, menoh::Dtype::Float)
        .unwrap();

    let vpt = vpt_builder
        .build_variable_profile_table(&model_data)
        .unwrap();

    // Attach buffer to input variable.
    // This is not necessary operation.
    // Internal buffer is automatically generated by model.
    let mut hen_im = to_input_vec(image::open(HEN_IMAGE_PATH).unwrap());
    let mut buffer = menoh::Buffer::new(&mut hen_im);

    let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
    model_builder
        .attach_external_buffer(CONV1_1_IN_NAME, &buffer, &vpt)
        .unwrap();

    let mut model = model_builder
        .build_model(&model_data, "mkldnn", "")
        .unwrap();

    model.run().unwrap();

    print_top_category(
        model.get_internal_buffer::<f32>(SOFTMAX_OUT_NAME).unwrap(),
        &categories,
    );

    // update buffer by cat image.
    let cat_im = to_input_vec(image::open(CAT_IMAGE_PATH).unwrap());
    buffer.update(&cat_im).unwrap();

    model.run().unwrap();
    print_top_category(
        model.get_internal_buffer::<f32>(SOFTMAX_OUT_NAME).unwrap(),
        &categories,
    );
}
