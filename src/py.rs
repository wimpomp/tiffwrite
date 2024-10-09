use pyo3::prelude::*;
use crate::{IJTiffFile, Tag};
use num::{Complex, Rational32, FromPrimitive};
use numpy::{PyReadonlyArray2, PyArrayMethods};


#[pyclass(subclass)]
#[pyo3(name = "Tag")]
#[derive(Clone, Debug)]
struct PyTag {
    tag: Tag
}

#[pymethods]
impl PyTag {
    #[staticmethod]
    fn byte(code: u16, byte: Vec<u8>) -> Self {
        PyTag { tag: Tag::byte(code, byte) }
    }

    #[staticmethod]
    fn ascii(code: u16, ascii: &str) -> Self {
        PyTag { tag: Tag::ascii(code, ascii) }
    }

    #[staticmethod]
    fn short(code: u16, short: Vec<u16>) -> Self {
        PyTag { tag: Tag::short(code, short) }
    }

    #[staticmethod]
    fn long(code: u16, long: Vec<u32>) -> Self {
        PyTag { tag: Tag::long(code, long) }
    }

    #[staticmethod]
    fn rational(code: u16, rational: Vec<f64>) -> Self {
        PyTag { tag: Tag::rational(code, rational.into_iter().map(|x| Rational32::from_f64(x).unwrap()).collect()) }
    }

    #[staticmethod]
    fn sbyte(code: u16, sbyte: Vec<i8>) -> Self {
        PyTag { tag: Tag::sbyte(code, sbyte) }
    }

    #[staticmethod]
    fn sshort(code: u16, sshort: Vec<i16>) -> Self {
        PyTag { tag: Tag::sshort(code, sshort) }
    }

    #[staticmethod]
    fn slong(code: u16, slong: Vec<i32>) -> Self {
        PyTag { tag: Tag::slong(code, slong) }
    }

    #[staticmethod]
    fn srational(code: u16, srational: Vec<f64>) -> Self {
        PyTag { tag: Tag::srational(code, srational.into_iter().map(|x| Rational32::from_f64(x).unwrap()).collect()) }
    }

    #[staticmethod]
    fn float(code: u16, float: Vec<f32>) -> Self {
        PyTag { tag: Tag::float(code, float) }
    }

    #[staticmethod]
    fn double(code: u16, double: Vec<f64>) -> Self {
        PyTag { tag: Tag::double(code, double) }
    }

    #[staticmethod]
    fn ifd(code: u16, ifd: Vec<u32>) -> Self {
        PyTag { tag: Tag::ifd(code, ifd) }
    }

    #[staticmethod]
    fn unicode(code: u16, unicode: &str) -> Self {
        PyTag { tag: Tag::unicode(code, unicode) }
    }

    #[staticmethod]
    fn complex(code: u16, complex: Vec<(f32, f32)>) -> Self {
        PyTag { tag: Tag::complex(code, complex.into_iter().map(|(x, y)| Complex { re: x, im: y }).collect()) }
    }

    #[staticmethod]
    fn long8(code: u16, long8: Vec<u64>) -> Self {
        PyTag { tag: Tag::long8(code, long8) }
    }

    #[staticmethod]
    fn slong8(code: u16, slong8: Vec<i64>) -> Self {
        PyTag { tag: Tag::slong8(code, slong8) }
    }

    #[staticmethod]
    fn ifd8(code: u16, ifd8: Vec<u64>) -> Self {
        PyTag { tag: Tag::ifd8(code, ifd8) }
    }

    fn count(&self) -> u64 {
        self.tag.count()
    }
}


#[pyclass(subclass)]
#[pyo3(name = "IJTiffFile")]
#[derive(Debug)]
struct PyIJTiffFile {
    ijtifffile: Option<IJTiffFile>
}

#[pymethods]
impl PyIJTiffFile {
    #[new]
    fn new(path: &str, shape: (usize, usize, usize)) -> PyResult<Self> {
        Ok(PyIJTiffFile { ijtifffile: Some(IJTiffFile::new(path, shape)?) } )
    }

    fn with_colors(&mut self, colors: (u8, u8, u8)) -> Self {
        todo!()
    }

    fn with_colormap(&mut self, colormap: Vec<(u8, u8, u8)>) -> Self {
        todo!()
    }

    fn with_px_size(&mut self, pxsize: f64) -> Self {
        todo!()
    }

    fn with_delta_z(&mut self, delta_z: f64) -> Self {
        todo!()
    }

    fn with_time_interval(&mut self, time_interval: f64) -> Self {
        todo!()
    }

    fn with_comments(&mut self, comments: String) -> Self {
        todo!()
    }

    fn append_extra_tag(&mut self, tag: PyTag) {
        if let Some(ijtifffile) = self.ijtifffile.as_mut() {
            if let Some(extra_tags) = ijtifffile.extra_tags.as_mut() {
                extra_tags.push(tag.tag);
            } else {
                ijtifffile.extra_tags = Some(vec![tag.tag]);
            }
        }
    }

    fn extend_extra_tags(&mut self, tags: Vec<PyTag>) {
        if let Some(ijtifffile) = self.ijtifffile.as_mut() {
            if let Some(extra_tags) = ijtifffile.extra_tags.as_mut() {
                extra_tags.extend(tags.into_iter().map(|x| x.tag));
            } else {
                ijtifffile.extra_tags = Some(tags.into_iter().map(|x| x.tag).collect());
            }
        }
    }

    fn close(&mut self) -> PyResult<()> {
        self.ijtifffile.take();
        Ok(())
    }
}


macro_rules! impl_save {
    ($T:ty, $t:ident) => {
        #[pymethods]
        impl PyIJTiffFile {
            fn $t(&mut self, frame: PyReadonlyArray2<$T>, c: usize, t: usize, z: usize,
                extra_tags: Option<Vec<PyTag>>) -> PyResult<()> {
                let extra_tags = if let Some(extra_tags) = extra_tags {
                    Some(extra_tags.into_iter().map(|x| x.tag).collect())
                } else {
                    None
                };
                if let Some(ijtifffile) = self.ijtifffile.as_mut() {
                    ijtifffile.save(frame.to_owned_array(), c, t, z, extra_tags)?;
                }
                Ok(())
            }
        }
    };
}

impl_save!(u8, save_u8);
impl_save!(u16, save_u16);
impl_save!(u32, save_u32);
impl_save!(u64, save_u64);
impl_save!(i8, save_i8);
impl_save!(i16, save_i16);
impl_save!(i32, save_i32);
impl_save!(i64, save_i64);
impl_save!(f32, save_f32);
impl_save!(f64, save_f64);


#[pymodule]
#[pyo3(name = "tiffwrite_rs")]
fn tiffwrite_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTag>()?;
    m.add_class::<PyIJTiffFile>()?;
    Ok(())
}
