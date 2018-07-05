//! # Menoh-rs
//! **A simple wrapper of [Menoh](https://github.com/pfnet-research/menoh)**
//!
//! # Examples
//! ```rust,ignore
//! extern crate menoh;
//!
//! // parse ONNX file
//! let model_data = menoh::ModelData::new(MODEL_PATH).unwrap();
//!
//! let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
//!
//! let input_dims = vec![
//!     INPUT_BATCH_SIZE,
//!     INPUT_CHANNEL_NUM,
//!     INPUT_HEIGHT,
//!     INPUT_WIDTH,
//! ];
//!
//! // register profile
//! vpt_builder
//!     .add_input_profile(CONV1_1_IN_NAME, menoh::Dtype::Float, &input_dims)
//!     .unwrap();
//!
//! vpt_builder
//!     .add_output_profile(SOFTMAX_OUT_NAME, menoh::Dtype::Float)
//!     .unwrap();
//!
//! let vpt = vpt_builder
//!     .build_variable_profile_table(&model_data)
//!     .unwrap();
//!
//! // Attach buffer to input variable.
//! // This is not necessary operation.
//! // Internal buffer is automatically generated by model.
//! let mut hen_im = to_input_vec(image::open(HEN_IMAGE_PATH).unwrap());
//! let mut buffer = menoh::Buffer::new(&mut hen_im);
//!
//! let mut model_builder = mehoh::ModelBuilder::new(&vpt).unwrap();
//! model_builder
//!     .attach_external_buffer(CONV1_1_IN_NAME, &buffer, &vpt)
//!     .unwrap();
//!
//! let mut model = model_builder
//!     .build_model(&model_data, "mkldnn", "")
//!     .unwrap();
//!
//! // run model inference
//! model.run().unwrap();
//! ```
//!

extern crate libc;

pub mod ffi;

mod buffer;
mod dtype;
mod error;
mod model;
mod model_data;
mod variable_profile;

pub use buffer::Buffer;
pub use dtype::{Dtype, DtypeCompatible};
pub use model::{Model, ModelBuilder};
pub use model_data::ModelData;
pub use variable_profile::{VariableProfile, VariableProfileTable, VariableProfileTableBuilder};
