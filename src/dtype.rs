use std;

use ffi;

#[derive(Clone, Copy)]
pub enum Dtype {
    Float,
}

pub trait DtypeCompatible: 'static + Clone + Copy {}

impl DtypeCompatible for f32 {}

impl Dtype {
    pub fn value(&self) -> ffi::menoh_dtype {
        match *self {
            Dtype::Float => ffi::menoh_dtype_float,
        }
    }

    pub fn from(dtype: ffi::menoh_dtype) -> Self {
        match dtype {
            ffi::menoh_dtype_float => Dtype::Float,
            _ => unreachable!(),
        }
    }

    pub fn type_id(&self) -> std::any::TypeId {
        match *self {
            Dtype::Float => std::any::TypeId::of::<f32>(),
        }
    }

    pub fn is_compatible<T: DtypeCompatible>(&self) -> bool {
        self.type_id() == std::any::TypeId::of::<T>()
    }
}
