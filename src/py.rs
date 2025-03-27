use crate::{Colors, Compression, IJTiffFile, Tag};
use ndarray::s;
use num::{Complex, FromPrimitive, Rational32};
use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(subclass)]
#[pyo3(name = "Tag")]
#[derive(Clone, Debug)]
struct PyTag {
    tag: Tag,
}

/// Tiff tag, use one of the constructors to get a tag of a specific type
#[pymethods]
impl PyTag {
    #[staticmethod]
    fn byte(code: u16, byte: Vec<u8>) -> Self {
        PyTag {
            tag: Tag::byte(code, &byte),
        }
    }

    #[staticmethod]
    fn ascii(code: u16, ascii: &str) -> Self {
        PyTag {
            tag: Tag::ascii(code, ascii),
        }
    }

    #[staticmethod]
    fn short(code: u16, short: Vec<u16>) -> Self {
        PyTag {
            tag: Tag::short(code, &short),
        }
    }

    #[staticmethod]
    fn long(code: u16, long: Vec<u32>) -> Self {
        PyTag {
            tag: Tag::long(code, &long),
        }
    }

    #[staticmethod]
    fn rational(code: u16, rational: Vec<f64>) -> Self {
        PyTag {
            tag: Tag::rational(
                code,
                &rational
                    .into_iter()
                    .map(|x| Rational32::from_f64(x).unwrap())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    #[staticmethod]
    fn sbyte(code: u16, sbyte: Vec<i8>) -> Self {
        PyTag {
            tag: Tag::sbyte(code, &sbyte),
        }
    }

    #[staticmethod]
    fn sshort(code: u16, sshort: Vec<i16>) -> Self {
        PyTag {
            tag: Tag::sshort(code, &sshort),
        }
    }

    #[staticmethod]
    fn slong(code: u16, slong: Vec<i32>) -> Self {
        PyTag {
            tag: Tag::slong(code, &slong),
        }
    }

    #[staticmethod]
    fn srational(code: u16, srational: Vec<f64>) -> Self {
        PyTag {
            tag: Tag::srational(
                code,
                &srational
                    .into_iter()
                    .map(|x| Rational32::from_f64(x).unwrap())
                    .collect::<Vec<_>>(),
            ),
        }
    }

    #[staticmethod]
    fn float(code: u16, float: Vec<f32>) -> Self {
        PyTag {
            tag: Tag::float(code, &float),
        }
    }

    #[staticmethod]
    fn double(code: u16, double: Vec<f64>) -> Self {
        PyTag {
            tag: Tag::double(code, &double),
        }
    }

    #[staticmethod]
    fn ifd(code: u16, ifd: Vec<u32>) -> Self {
        PyTag {
            tag: Tag::ifd(code, &ifd),
        }
    }

    #[staticmethod]
    fn unicode(code: u16, unicode: &str) -> Self {
        PyTag {
            tag: Tag::unicode(code, unicode),
        }
    }

    #[staticmethod]
    fn complex(code: u16, complex: Vec<(f32, f32)>) -> Self {
        PyTag {
            tag: Tag::complex(
                code,
                &complex
                    .into_iter()
                    .map(|(x, y)| Complex { re: x, im: y })
                    .collect::<Vec<_>>(),
            ),
        }
    }

    #[staticmethod]
    fn long8(code: u16, long8: Vec<u64>) -> Self {
        PyTag {
            tag: Tag::long8(code, &long8),
        }
    }

    #[staticmethod]
    fn slong8(code: u16, slong8: Vec<i64>) -> Self {
        PyTag {
            tag: Tag::slong8(code, &slong8),
        }
    }

    #[staticmethod]
    fn ifd8(code: u16, ifd8: Vec<u64>) -> Self {
        PyTag {
            tag: Tag::ifd8(code, &ifd8),
        }
    }

    /// get the number of values in the tag
    fn count(&self) -> u64 {
        self.tag.count()
    }
}

#[pyclass(subclass)]
#[pyo3(name = "IJTiffFile")]
#[derive(Debug)]
struct PyIJTiffFile {
    ijtifffile: Option<IJTiffFile>,
}

#[pymethods]
impl PyIJTiffFile {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(PyIJTiffFile {
            ijtifffile: Some(IJTiffFile::new(path)?),
        })
    }

    /// set zstd compression level: -7 ..= 22
    fn set_compression(&mut self, compression: i32, level: i32) -> PyResult<()> {
        let c = match compression {
            50000 => Compression::Zstd(level.clamp(-7, 22)),
            8 => Compression::Deflate,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown compression {}",
                    compression
                )))
            }
        };
        if let Some(ref mut ijtifffile) = self.ijtifffile {
            ijtifffile.set_compression(c)
        }
        Ok(())
    }

    #[getter]
    fn get_colors(&self) -> PyResult<Option<Vec<Vec<u8>>>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            if let Colors::Colors(colors) = &ijtifffile.colors {
                return Ok(Some(colors.to_owned()));
            }
        }
        Ok(None)
    }

    #[setter]
    fn set_colors(&mut self, colors: PyReadonlyArray2<u8>) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            let a = colors.to_owned_array();
            ijtifffile.colors = Colors::Colors(
                (0..a.shape()[0])
                    .map(|i| Vec::from(a.slice(s![i, ..]).as_slice().unwrap()))
                    .collect(),
            );
        }
        Ok(())
    }

    #[getter]
    fn get_colormap(&mut self) -> PyResult<Option<Vec<Vec<u8>>>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            if let Colors::Colormap(colormap) = &ijtifffile.colors {
                return Ok(Some(colormap.to_owned()));
            }
        }
        Ok(None)
    }

    #[setter]
    fn set_colormap(&mut self, colormap: PyReadonlyArray2<u8>) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            let a = colormap.to_owned_array();
            ijtifffile.colors = Colors::Colormap(
                (0..a.shape()[0])
                    .map(|i| Vec::from(a.slice(s![i, ..]).as_slice().unwrap()))
                    .collect(),
            );
        }
        Ok(())
    }

    #[getter]
    fn get_px_size(&self) -> PyResult<Option<f64>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            Ok(ijtifffile.px_size)
        } else {
            Ok(None)
        }
    }

    #[setter]
    fn set_px_size(&mut self, px_size: f64) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            ijtifffile.px_size = Some(px_size);
        }
        Ok(())
    }

    #[getter]
    fn get_delta_z(&self) -> PyResult<Option<f64>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            Ok(ijtifffile.delta_z)
        } else {
            Ok(None)
        }
    }

    #[setter]
    fn set_delta_z(&mut self, delta_z: f64) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            ijtifffile.delta_z = Some(delta_z);
        }
        Ok(())
    }

    #[getter]
    fn get_time_interval(&self) -> PyResult<Option<f64>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            Ok(ijtifffile.time_interval)
        } else {
            Ok(None)
        }
    }

    #[setter]
    fn set_time_interval(&mut self, time_interval: f64) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            ijtifffile.time_interval = Some(time_interval);
        }
        Ok(())
    }

    #[getter]
    fn get_comment(&self) -> PyResult<Option<String>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            Ok(ijtifffile.comment.clone())
        } else {
            Ok(None)
        }
    }

    #[setter]
    fn set_comment(&mut self, comment: &str) -> PyResult<()> {
        if let Some(ijtifffile) = &mut self.ijtifffile {
            ijtifffile.comment = Some(String::from(comment));
        }
        Ok(())
    }

    #[pyo3(signature = (tag, czt=None))]
    fn append_extra_tag(&mut self, tag: PyTag, czt: Option<(usize, usize, usize)>) {
        if let Some(ijtifffile) = self.ijtifffile.as_mut() {
            if let Some(extra_tags) = ijtifffile.extra_tags.get_mut(&czt) {
                extra_tags.push(tag.tag)
            }
        }
    }

    #[pyo3(signature = (czt=None))]
    fn get_tags(&self, czt: Option<(usize, usize, usize)>) -> PyResult<Vec<PyTag>> {
        if let Some(ijtifffile) = &self.ijtifffile {
            if let Some(extra_tags) = ijtifffile.extra_tags.get(&czt) {
                let v = extra_tags
                    .iter()
                    .map(|tag| PyTag {
                        tag: tag.to_owned(),
                    })
                    .collect();
                return Ok(v);
            }
        }
        Ok(Vec::new())
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
            fn $t(
                &mut self,
                frame: PyReadonlyArray2<$T>,
                c: usize,
                t: usize,
                z: usize,
            ) -> PyResult<()> {
                if let Some(ijtifffile) = self.ijtifffile.as_mut() {
                    ijtifffile.save(frame.as_array(), c, t, z)?;
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
