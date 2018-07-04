use dtype::DtypeCompatible;
use error::Error;

pub struct Buffer<T>
where
    T: DtypeCompatible,
{
    data: Vec<T>,
}

impl<T> Buffer<T>
where
    T: DtypeCompatible,
{
    pub fn new(len: usize) -> Self {
        Buffer {
            data: vec![Default::default(); len],
        }
    }

    pub fn from_slice(data: &[T]) -> Self {
        Buffer {
            data: data.to_vec(),
        }
    }

    pub fn update(&mut self, data: &[T]) -> Result<(), Error> {
        if self.data.len() != data.len() {
            return Err(Error::InvalidBufferSize);
        }

        self.data.copy_from_slice(data);
        Ok(())
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
