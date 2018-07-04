use dtype::DtypeCompatible;
use error::Error;

pub struct Buffer<'a, T>
where
    T: 'a + DtypeCompatible,
{
    data: &'a mut [T],
}

impl<'a, T> Buffer<'a, T>
where
    T: 'a + DtypeCompatible,
{
    pub fn new(data: &'a mut [T]) -> Self {
        Buffer { data }
    }

    pub fn update(&mut self, data: &[T]) -> Result<(), Error> {
        if self.data.len() != data.len() {
            return Err(Error::InvalidBufferSize);
        }

        self.data.copy_from_slice(data);
        Ok(())
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}
