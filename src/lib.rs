extern crate libc;

pub mod ffi;

pub mod error;
pub mod model;
pub mod model_data;
pub mod variable_profile;

pub use model::{Model, ModelBuilder};
pub use model_data::ModelData;
pub use variable_profile::{Dtype, VariableProfileTable, VariableProfileTableBuilder};
