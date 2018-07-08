//! # Menoh-rs
//! **A simple wrapper of [Menoh](https://github.com/pfnet-research/menoh)**
//!
//! # Examples
//! ```rust
//! extern crate menoh;
//! extern crate image;
//!
//! use image::GenericImage;
//!
//! const INPUT_VARIABLE_NAME: &str = "139900320569040";
//! const OUTPUT_VARIABLE_NAME: &str = "139898462888656";
//!
//! const INPUT_BATCH_SIZE: i32 = 1;
//! const INPUT_CHANNEL_NUM: i32 = 1;
//! const INPUT_WIDTH: i32 = 28;
//! const INPUT_HEIGHT: i32 = 28;
//!
//! const MODEL_PATH: &str = "tests/resource/mnist.onnx";
//! const IMAGE_PATH: &str = "tests/resource/1.png";
//!
//! fn main() {
//!     // load category file
//!     let model_data = menoh::ModelData::new(MODEL_PATH).unwrap();
//!     let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
//!
//!     let input_dims = vec![
//!         INPUT_BATCH_SIZE,
//!         INPUT_CHANNEL_NUM,
//!         INPUT_HEIGHT,
//!         INPUT_WIDTH,
//!     ];
//!     // register profile
//!     vpt_builder
//!         .add_input_profile(INPUT_VARIABLE_NAME, menoh::Dtype::Float, &input_dims)
//!         .unwrap();
//!     vpt_builder
//!         .add_output_profile(OUTPUT_VARIABLE_NAME, menoh::Dtype::Float)
//!         .unwrap();
//!
//!     let vpt = vpt_builder
//!         .build_variable_profile_table(&model_data)
//!         .unwrap();
//!
//!     // Create Buffer
//!     let mut im = to_input_vec(image::open(IMAGE_PATH).unwrap());
//!     let buffer = menoh::Buffer::new(&mut im);
//!
//!     // Attach buffer to input variable.
//!     // This is not necessary operation.
//!     // Internal buffer is automatically generated by model.
//!     let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
//!     model_builder
//!         .attach_external_buffer(INPUT_VARIABLE_NAME, &buffer, &vpt)
//!         .unwrap();
//!
//!     let mut model = model_builder
//!         .build_model(&model_data, menoh::Backend::MKL_DNN, "")
//!         .unwrap();
//!
//!     model.run().unwrap();
//!
//!     println!(
//!         "{:?}",
//!         model
//!             .get_internal_buffer::<f32>(OUTPUT_VARIABLE_NAME)
//!             .unwrap()
//!             .as_slice()
//!     );
//! }
//!
//! // ###### Just create input data below here ######
//!
//! fn resize_im(mut im: image::DynamicImage, width: u32, height: u32) -> image::DynamicImage {
//!     let im_w = im.width();
//!     let im_h = im.height();
//!     let shortest_edge = std::cmp::min(im_h, im_w);
//!     let im = im.crop(
//!         (im_w - shortest_edge) / 2,
//!         (im_h - shortest_edge) / 2,
//!         shortest_edge,
//!         shortest_edge,
//!     );
//!     im.resize_exact(width, height, image::FilterType::Nearest)
//! }
//!
//! fn reorder_to_chw(im: &image::DynamicImage) -> Vec<f32> {
//!     let (im_h, im_w) = (im.height(), im.width());
//!     let mut input_im: Vec<f32> = vec![Default::default(); (im_h * im_w) as usize];
//!     for h in 0..im_h {
//!         for w in 0..im_w {
//!             input_im[(w * im_h + h) as usize] = im.get_pixel(h as u32, w as u32)[0] as f32;
//!         }
//!     }
//!     input_im
//! }
//!
//! fn to_input_vec(im: image::DynamicImage) -> Vec<f32> {
//!     let im = resize_im(im, INPUT_WIDTH as _, INPUT_HEIGHT as _);
//!     let im = im.grayscale();
//!     reorder_to_chw(&im)
//! }
//! ```
//!

extern crate libc;

#[cfg(test)]
#[macro_use]
extern crate matches;

pub mod ffi;

mod backend;
mod buffer;
mod dtype;
mod error;
mod model;
mod model_data;
mod variable_profile;

pub use backend::Backend;
pub use buffer::Buffer;
pub use dtype::{Dtype, DtypeCompatible};
pub use error::Error;
pub use model::{Model, ModelBuilder};
pub use model_data::ModelData;
pub use variable_profile::{VariableProfile, VariableProfileTable, VariableProfileTableBuilder};
