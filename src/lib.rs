#[cfg(not(feature = "nopython"))]
mod py;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use anyhow::Result;
use fraction::Fraction;
use num::{Complex, Zero};
use num::complex::ComplexFloat;
use ndarray::{s, Array2};
use num::traits::ToBytes;
use std::hash::{DefaultHasher, Hash, Hasher};
use chrono::Utc;


const TAG_SIZE: usize = 20;
const OFFSET_SIZE: usize = 8;
const OFFSET: u64 = 16;
const COMPRESSION: u16  = 1;


#[derive(Clone, Debug)]
struct IFD {
    tags: Vec<Tag>
}

impl IFD {
    pub fn new() -> Self {
        IFD { tags: Vec::new() }
    }

    fn push_tag(&mut self, tag: Tag) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    fn extend_tags(&mut self, tags: Vec<Tag>) {
        for tag in tags {
            self.push_tag(tag);
        }
    }

    fn write(&mut self, ijtifffile: &mut IJTiffFile, where_to_write_offset: u64) -> Result<u64> {
        self.tags.sort();
        ijtifffile.file.seek(SeekFrom::End(0))?;
        if ijtifffile.file.stream_position()? % 2 == 1 {
            ijtifffile.file.write(&[0])?;
        }
        let offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write(&(self.tags.len() as u64).to_le_bytes())?;

        for tag in self.tags.iter_mut() {
            tag.write_tag(ijtifffile)?;
        }
        let where_to_write_next_ifd_offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write(&vec![0u8; OFFSET_SIZE])?;
        for tag in self.tags.iter() {
            tag.write_data(ijtifffile)?;
        }
        ijtifffile.file.seek(SeekFrom::Start(where_to_write_offset))?;
        ijtifffile.file.write(&offset.to_le_bytes())?;
        Ok(where_to_write_next_ifd_offset)
    }
}


#[derive(Clone, Debug, Eq)]
pub struct Tag {
    code: u16,
    bytes: Vec<u8>,
    ttype: u16,
    offset: u64
}

impl PartialOrd<Self> for Tag {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tag {
    fn cmp(&self, other: &Self) -> Ordering {
        self.code.cmp(&other.code)
    }
}

impl PartialEq for Tag {
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Tag {
    pub fn new(code: u16, bytes: Vec<u8>, ttype: u16) -> Self {
        Tag { code, bytes, ttype, offset: 0 }
    }

    pub fn byte(code: u16, byte: Vec<u8>) -> Self {
        Tag::new(code, byte, 1)
    }

    pub fn ascii(code: u16, ascii: &str) -> Self {
        let mut bytes = ascii.as_bytes().to_vec();
        bytes.push(0);
        Tag::new(code, bytes, 2)
    }

    pub fn short(code: u16, short: Vec<u16>) -> Self {
        Tag::new(code, short.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 3)
    }

    pub fn long(code: u16, long: Vec<u32>) -> Self {
        Tag::new(code, long.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 4)
    }

    pub fn rational(code: u16, rational: Vec<Fraction>) -> Self {
        Tag::new(code, rational.into_iter().map(|x|
            u32::try_from(*x.denom().unwrap()).unwrap().to_le_bytes().into_iter().chain(
                u32::try_from(*x.numer().unwrap()).unwrap().to_le_bytes()).collect::<Vec<_>>()
        ).flatten().collect(), 5)
    }

    pub fn sbyte(code: u16, sbyte: Vec<i8>) -> Self {
        Tag::new(code, sbyte.iter().map(|x| x.to_le_bytes()).flatten().collect(), 6)
    }

    pub fn sshort(code: u16, sshort: Vec<i16>) -> Self {
        Tag::new(code, sshort.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 8)
    }

    pub fn slong(code: u16, slong: Vec<i32>) -> Self {
        Tag::new(code, slong.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 9)
    }

    pub fn srational(code: u16, srational: Vec<Fraction>) -> Self {
        Tag::new(code, srational.into_iter().map(|x|
            i32::try_from(*x.denom().unwrap()).unwrap().to_le_bytes().into_iter().chain(
                i32::try_from(*x.numer().unwrap()).unwrap().to_le_bytes()).collect::<Vec<_>>()
        ).flatten().collect(), 10)
    }

    pub fn float(code: u16, float: Vec<f32>) -> Self {
        Tag::new(code, float.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 11)
    }

    pub fn double(code: u16, double: Vec<f64>) -> Self {
        Tag::new(code, double.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 12)
    }

    pub fn ifd(code: u16, ifd: Vec<u32>) -> Self {
        Tag::new(code, ifd.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 13)
    }

    pub fn unicode(code: u16, unicode: &str) -> Self {
        let mut bytes: Vec<u8> = unicode.encode_utf16().map(|x| x.to_le_bytes()).flatten().collect();
        bytes.push(0);
        Tag::new(code, bytes, 14)
    }

    pub fn complex(code: u16, complex: Vec<Complex<f32>>) -> Self {
        Tag::new(code, complex.into_iter().map(|x|
            x.re().to_le_bytes().into_iter().chain(x.im().to_le_bytes()).collect::<Vec<_>>()
        ).flatten().collect(), 15)
    }

    pub fn long8(code: u16, long8: Vec<u64>) -> Self {
        Tag::new(code, long8.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 16)
    }

    pub fn slong8(code: u16, slong8: Vec<i64>) -> Self {
        Tag::new(code, slong8.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 17)
    }

    pub fn ifd8(code: u16, ifd8: Vec<u64>) -> Self {
        Tag::new(code, ifd8.into_iter().map(|x| x.to_le_bytes()).flatten().collect(), 18)
    }

    pub fn count(&self) -> u64 {
        let c = match self.ttype {
            1 => self.bytes.len(),  // BYTE
            2 => self.bytes.len(),  // ASCII
            3 => self.bytes.len() / 2,  // SHORT
            4 => self.bytes.len() / 4,  // LONG
            5 => self.bytes.len() / 8,  // RATIONAL
            6 => self.bytes.len(),  // SBYTE
            7 => self.bytes.len(),  // UNDEFINED
            8 => self.bytes.len() / 2,  // SSHORT
            9 => self.bytes.len() / 4,  // SLONG
            10 => self.bytes.len() / 8,  // SRATIONAL
            11 => self.bytes.len() / 4,  // FLOAT
            12 => self.bytes.len() / 8,  // DOUBLE
            13 => self.bytes.len() / 4,  // IFD
            14 => self.bytes.len() / 2,  // UNICODE
            15 => self.bytes.len() / 8,  // COMPLEX
            16 => self.bytes.len() / 8,  // LONG8
            17 => self.bytes.len() / 8,  // SLONG8
            18 => self.bytes.len() / 8,  // IFD8
            _ => self.bytes.len(),
        };
        c as u64
    }

    fn write_tag(&mut self, ijtifffile: &mut IJTiffFile) -> Result<()> {
        self.offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write(&self.code.to_le_bytes())?;
        ijtifffile.file.write(&self.ttype.to_le_bytes())?;
        ijtifffile.file.write(&self.count().to_le_bytes())?;
        if self.bytes.len() <= OFFSET_SIZE {
            ijtifffile.file.write(&self.bytes)?;
            for _ in self.bytes.len()..OFFSET_SIZE {
                ijtifffile.file.write(&[0])?;
            }
        } else {
            ijtifffile.file.write(&vec![0u8; OFFSET_SIZE])?;
        }
        Ok(())
    }

    fn write_data(&self, ijtifffile: &mut IJTiffFile) -> Result<()> {
        if self.bytes.len() > OFFSET_SIZE {
            ijtifffile.file.seek(SeekFrom::End(0))?;
            let offset = ijtifffile.write(&self.bytes)?;
            ijtifffile.file.seek(SeekFrom::Start(
                self.offset + (TAG_SIZE - OFFSET_SIZE) as u64))?;
            ijtifffile.file.write(&offset.to_le_bytes())?;
            if ijtifffile.file.stream_position()? % 2 == 1 {
                ijtifffile.file.write(&[0u8])?;
            }
        }
        Ok(())
    }
}


#[derive(Clone, Debug)]
struct Frame {
    tilebyteoffsets: Vec<u64>,
    tilebytecounts: Vec<u64>,
    image_width: u32,
    image_length: u32,
    bits_per_sample: u16,
    sample_format: u16,
    tile_width: u16,
    tile_length: u16,
    extra_tags: Vec<Tag>
}

impl Frame {
    fn new(
        tilebyteoffsets: Vec<u64>, tilebytecounts: Vec<u64>, image_width: u32, image_length: u32,
        bits_per_sample: u16, sample_format: u16, tile_width: u16, tile_length: u16
    ) -> Self {
        Frame {
            tilebyteoffsets, tilebytecounts, image_width, image_length, bits_per_sample,
            sample_format, tile_width, tile_length, extra_tags: Vec::new()
        }
    }
}


pub trait Bytes {
    const BITS_PER_SAMPLE: u16;
    const SAMPLE_FORMAT: u16;

    fn bytes(&self) -> Vec<u8>;
}


macro_rules! bytes_impl {
    ($T:ty, $bits_per_sample:expr, $sample_format:expr) => {
        impl Bytes for $T {
            const BITS_PER_SAMPLE: u16 = $bits_per_sample;
            const SAMPLE_FORMAT: u16 = $sample_format;

            #[inline]
            fn bytes(&self) -> Vec<u8>
            {
                self.to_le_bytes().to_vec()
            }
        }
    };
}

bytes_impl!(u8, 8, 1);
bytes_impl!(u16, 16, 1);
bytes_impl!(u32, 32, 1);
bytes_impl!(u64, 64, 1);
bytes_impl!(u128, 128, 1);
#[cfg(target_pointer_width = "64")]
bytes_impl!(usize, 64, 1);
#[cfg(target_pointer_width = "32")]
bytes_impl!(usize, 32, 1);

bytes_impl!(i8, 8, 2);
bytes_impl!(i16, 16, 2);
bytes_impl!(i32, 32, 2);
bytes_impl!(i64, 64, 2);
bytes_impl!(i128, 128, 2);
#[cfg(target_pointer_width = "64")]
bytes_impl!(isize, 64, 2);
#[cfg(target_pointer_width = "32")]
bytes_impl!(isize, 32, 2);
bytes_impl!(f32, 32, 3);
bytes_impl!(f64, 64, 3);


#[derive(Debug)]
pub struct IJTiffFile {
    file: File,
    frames: HashMap<(usize, u8), Frame>,
    hashes: HashMap<u64, u64>,
    pub shape: (usize, usize, usize),
    pub n_frames: usize,
    pub samples_per_pixel: u8,
    pub colormap: Option<Vec<u16>>,
    pub colors: Option<Vec<(u8, u8, u8)>>,
    pub comment: Option<String>,
    pub delta_z: Option<f64>,
    pub timeinterval: Option<f64>,
    pub extra_tags: Option<Vec<Tag>>
}

impl Drop for IJTiffFile {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            println!("Error closing IJTiffFile: {:?}", e);
        }
    }
}

impl IJTiffFile {
    pub fn new(path: &str, shape: (usize, usize, usize)) -> Result<Self> {
        let mut file = OpenOptions::new().create(true).truncate(true)
            .write(true).read(true).open(path)?;
        file.write(b"II")?;
        file.write(&43u16.to_le_bytes())?;
        file.write(&8u16.to_le_bytes())?;
        file.write(&0u16.to_le_bytes())?;
        file.write(&OFFSET.to_le_bytes())?;
        let colormap: Option<Vec<(u8, u8, u8)>> = None;
        let (spp, n_frames) = if let None = &colormap {
            (shape.0 as u8, shape.1 * shape.2)
        } else {
            (1, shape.0 * shape.1 * shape.2)
        };
        Ok(IJTiffFile { file, frames: HashMap::new(), hashes: HashMap::new(), shape, n_frames,
            samples_per_pixel: spp, colormap: None, colors: None, comment: None, delta_z: None,
            timeinterval: None, extra_tags: None } )
    }

    pub fn description(&self) -> String {
        let mut desc: String = String::from("ImageJ=1.11a");
        if let (None, None) = (self.colormap.as_ref(), self.colors.as_ref()) {
            desc += &format!("\nimages={}", self.shape.0);
            desc += &format!("\nslices={}", self.shape.1);
            desc += &format!("\nframes={}", self.shape.2);
        } else {
            desc += &format!("\nimages={}", self.shape.0 * self.shape.1 * self.shape.2);
            desc += &format!("\nchannels={}", self.shape.0);
            desc += &format!("\nslices={}", self.shape.1);
            desc += &format!("\nframes={}", self.shape.2);
        };
        if self.shape.0 == 1 {
            desc += "\nmode=grayscale";
        } else {
            desc += "\nmode=composite";
        }
        desc += "\nhyperstack=true\nloop=false\nunit=micron";
        if let Some(delta_z) = self.delta_z {
            desc += &format!("\nspacing={}", delta_z);
        }
        if let Some(timeinterval) = self.timeinterval {
            desc += &format!("\ninterval={}", timeinterval);
        }
        if let Some(comment) = &self.comment {
            desc += &format!("\ncomment={}", comment);
        }
        desc
    }

    fn get_frame_number(&self, c: usize, z: usize, t: usize) -> (usize, u8) {
        if let (None, None) = (self.colormap.as_ref(), self.colors.as_ref()) {
            (z + t * self.shape.1, c as u8)
        } else {
            (c + z * self.shape.0 + t * self.shape.0 * self.shape.1, 0)
        }
    }

    fn hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_check(&mut self, bytes: &Vec<u8>, offset: u64) -> Result<bool> {
        let current_offset = self.file.stream_position()?;
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; bytes.len()];
        self.file.read_exact(&mut buffer)?;
        let same = bytes == &buffer;
        self.file.seek(SeekFrom::Start(current_offset))?;
        Ok(same)
    }

    fn write(&mut self, bytes: &Vec<u8>) -> Result<u64> {
        let hash = IJTiffFile::hash(&bytes);
        if self.hashes.contains_key(&hash) && self.hash_check(&bytes, *self.hashes.get(&hash).unwrap())? {
            Ok(*self.hashes.get(&hash).unwrap())
        } else {
            if self.file.stream_position()? % 2 == 1 {
                self.file.write(&[0])?;
            }
            let offset = self.file.stream_position()?;
            self.hashes.insert(hash, offset);
            self.file.write(&bytes)?;
            Ok(offset)
        }
    }

    pub fn save<T: Bytes + Clone + Zero>(&mut self, frame: Array2<T>, c: usize, z: usize, t: usize,
                extra_tags: Option<Vec<Tag>>) -> Result<()> {
        self.compress_frame(frame, c, z, t, extra_tags);
        Ok(())
    }

    fn compress_frame<T: Bytes + Clone + Zero>(&mut self, frame: Array2<T>, c: usize, z: usize, t: usize,
                                                     extra_tags: Option<Vec<Tag>>) {
        let image_width = frame.shape()[0] as u32;
        let image_length = frame.shape()[1] as u32;
        let mut tilebyteoffsets = Vec::new();
        let mut tilebytecounts = Vec::new();
        let tiles = IJTiffFile::tile(frame.reversed_axes(), 64);
        for tile in tiles {
            let bytes: Vec<u8> = tile.map(|x| x.bytes()).into_iter().flatten().collect();
            tilebytecounts.push(bytes.len() as u64);
            tilebyteoffsets.push(self.write(&bytes).unwrap());
        }
        let mut frame = Frame::new(tilebyteoffsets, tilebytecounts, image_width, image_length,
                               T::BITS_PER_SAMPLE, T::SAMPLE_FORMAT, 64, 64);
        if let Some(tags) = extra_tags {
            for tag in tags {
                frame.extra_tags.push(tag);
            }
        }
        self.frames.insert(self.get_frame_number(c, z, t), frame);
    }

    fn tile<T: Clone + Zero>(frame: Array2<T>, size: usize) -> Vec<Array2<T>> {
        let shape = frame.shape();
        let mut tiles = Vec::new();
        let (n, m) = (shape[0] / size, shape[1] / size);
        for i in 0..n {
            for j in 0..m {
                tiles.push(frame.slice(
                    s![i * size..(i + 1) * size, j * size..(j + 1) * size]).to_owned());
            }
            if shape[1] % size != 0 {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(
                    s![.., ..shape[1] - m * size]
                ).assign(&frame.slice(s![i * size..(i + 1) * size, m * size..]));
                tiles.push(tile);
            }
        }
        if shape[0] % size != 0 {
            for j in 0..m {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(
                    s![..shape[0] - n * size, ..]
                ).assign(&frame.slice(s![n * size.., j * size..(j + 1) * size]));
                tiles.push(tile);
            }
            if shape[1] % size != 0 {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(
                    s![..shape[0] - n * size, ..shape[1] - m * size]
                ).assign(&frame.slice(s![n * size.., m * size..]));
                tiles.push(tile);
            }
        }
        tiles
    }

    fn get_colormap(&self, colormap: &Vec<u16>) -> Result<Vec<u16>> {
        todo!();
    }

    fn get_color(&self, colors: (u8, u8, u8)) -> Result<Vec<u16>> {
        todo!();
    }

    fn close(&mut self) -> Result<()> {
        let mut where_to_write_next_ifd_offset = OFFSET - OFFSET_SIZE as u64;
        let mut warn = false;
        for frame_number in 0..self.n_frames {
            if let Some(frame) = self.frames.get(&(frame_number, 0)) {
                let mut tilebyteoffsets = Vec::new();
                let mut tilebytecounts = Vec::new();
                let mut frame_count = 0;
                for channel in 0..self.samples_per_pixel {
                    if let Some(frame_n) = self.frames.get(&(frame_number, channel)) {
                        tilebyteoffsets.extend(frame_n.tilebyteoffsets.iter());
                        tilebytecounts.extend(frame_n.tilebytecounts.iter());
                        frame_count += 1;
                    } else {
                        warn = true;
                    }
                }
                let mut ifd = IFD::new();
                ifd.push_tag(Tag::long(256, vec![frame.image_width]));
                ifd.push_tag(Tag::long(257, vec![frame.image_length]));
                ifd.push_tag(Tag::short(258, vec![frame.bits_per_sample; frame_count]));
                ifd.push_tag(Tag::short(259, vec![COMPRESSION]));
                ifd.push_tag(Tag::ascii(270, &self.description()));
                ifd.push_tag(Tag::short(277, vec![frame_count as u16]));
                ifd.push_tag(Tag::ascii(305, "tiffwrite_rs"));
                ifd.push_tag(Tag::short(322, vec![frame.tile_width]));
                ifd.push_tag(Tag::short(323, vec![frame.tile_length]));
                ifd.push_tag(Tag::long8(324, tilebyteoffsets));
                ifd.push_tag(Tag::long8(325, tilebytecounts));
                ifd.push_tag(Tag::short(339, vec![frame.sample_format]));
                if frame_number == 0 {
                    if let Some(colormap) = &self.colormap {
                        ifd.push_tag(Tag::short(320, self.get_colormap(colormap)?));
                        ifd.push_tag(Tag::short(262, vec![3]));  // PhotometricInterpretation PHOTOMETRIC_PALETTE
                    } else if let None = self.colors {
                        ifd.push_tag(Tag::short(262, vec![1]));  // PhotometricInterpretation PHOTOMETRIC_PALETTE
                    }
                }
                if frame_number < self.samples_per_pixel as usize {
                    if let Some(color) = &self.colors {
                        ifd.push_tag(Tag::short(320, self.get_color(color[frame_number])?));
                        ifd.push_tag(Tag::short(262, vec![3]));  // PhotometricInterpretation PHOTOMETRIC_PALETTE
                    }
                }
                if let (None, None) = (&self.colormap, &self.colors) {
                    if self.shape.0 > 1 {
                        ifd.push_tag(Tag::short(284, vec![2]))
                    }
                }
                ifd.extend_tags(frame.extra_tags.to_owned());
                if let Some(extra_tags) = &self.extra_tags {
                    ifd.extend_tags(extra_tags.to_owned());
                }
                ifd.push_tag(Tag::ascii(306, &format!("{}", Utc::now().format("%Y:%m:%d %H:%M:%S"))));
                where_to_write_next_ifd_offset = ifd.write(self, where_to_write_next_ifd_offset)?;
            } else {
                warn = true;
            }
            if warn {
                println!("Some frames were not added to the tif file, either you forgot them, \
                    or an error occurred and the tif file was closed prematurely.")
            }
        }
        self.file.seek(SeekFrom::Start(where_to_write_next_ifd_offset))?;
        self.file.write(&0u64.to_le_bytes())?;
        Ok(())
    }
}