#[cfg(feature = "python")]
mod py;

use anyhow::Result;
use chrono::Utc;
use ndarray::{s, Array2};
use num::{traits::ToBytes, Complex, FromPrimitive, Rational32, Zero};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{copy, Read, Seek, SeekFrom, Write};
use std::time::Duration;
use std::{cmp::Ordering, collections::HashMap};
use std::{
    thread,
    thread::{sleep, JoinHandle},
};
use zstd::{stream::Encoder, DEFAULT_COMPRESSION_LEVEL};

const TAG_SIZE: usize = 20;
const OFFSET_SIZE: usize = 8;
const OFFSET: u64 = 16;
const COMPRESSION: u16 = 50000;

fn encode_all(source: Vec<u8>, level: i32) -> Result<Vec<u8>> {
    let mut result = Vec::<u8>::new();
    copy_encode(&*source, &mut result, level, source.len() as u64)?;
    Ok(result)
}

/// copy_encode from zstd crate, but let it include the content size in the zstd block header
fn copy_encode<R, W>(mut source: R, destination: W, level: i32, length: u64) -> Result<()>
where
    R: Read,
    W: Write,
{
    let mut encoder = Encoder::new(destination, level)?;
    encoder.include_contentsize(true)?;
    encoder.set_pledged_src_size(Some(length))?;
    copy(&mut source, &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

#[derive(Clone, Debug)]
struct IFD {
    tags: HashSet<Tag>,
}

impl IFD {
    pub fn new() -> Self {
        IFD {
            tags: HashSet::new(),
        }
    }

    fn write(&mut self, ijtifffile: &mut IJTiffFile, where_to_write_offset: u64) -> Result<u64> {
        let mut tags = self.tags.drain().collect::<Vec<_>>();
        tags.sort();
        ijtifffile.file.seek(SeekFrom::End(0))?;
        if ijtifffile.file.stream_position()? % 2 == 1 {
            ijtifffile.file.write(&[0])?;
        }
        let offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write(&(tags.len() as u64).to_le_bytes())?;

        for tag in tags.iter_mut() {
            tag.write_tag(ijtifffile)?;
        }
        let where_to_write_next_ifd_offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write(&vec![0u8; OFFSET_SIZE])?;
        for tag in tags.iter() {
            tag.write_data(ijtifffile)?;
        }
        ijtifffile
            .file
            .seek(SeekFrom::Start(where_to_write_offset))?;
        ijtifffile.file.write(&offset.to_le_bytes())?;
        Ok(where_to_write_next_ifd_offset)
    }
}

#[derive(Clone, Debug, Eq)]
pub struct Tag {
    code: u16,
    bytes: Vec<u8>,
    ttype: u16,
    offset: u64,
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

impl Hash for Tag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.code.hash(state);
    }
}

impl Tag {
    pub fn new(code: u16, bytes: Vec<u8>, ttype: u16) -> Self {
        Tag {
            code,
            bytes,
            ttype,
            offset: 0,
        }
    }

    pub fn byte(code: u16, value: &Vec<u8>) -> Self {
        Tag::new(code, value.to_owned(), 1)
    }

    pub fn ascii(code: u16, value: &str) -> Self {
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        Tag::new(code, bytes, 2)
    }

    pub fn short(code: u16, value: &Vec<u16>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            3,
        )
    }

    pub fn long(code: u16, value: &Vec<u32>) -> Self {
        Tag::new(
            code,
            value.into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            4,
        )
    }

    pub fn rational(code: u16, value: &Vec<Rational32>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| {
                    u32::try_from(*x.denom())
                        .unwrap()
                        .to_le_bytes()
                        .into_iter()
                        .chain(u32::try_from(*x.numer()).unwrap().to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect(),
            5,
        )
    }

    pub fn sbyte(code: u16, value: &Vec<i8>) -> Self {
        Tag::new(
            code,
            value.iter().map(|x| x.to_le_bytes()).flatten().collect(),
            6,
        )
    }

    pub fn sshort(code: u16, value: &Vec<i16>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            8,
        )
    }

    pub fn slong(code: u16, value: &Vec<i32>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            9,
        )
    }

    pub fn srational(code: u16, value: &Vec<Rational32>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| {
                    i32::try_from(*x.denom())
                        .unwrap()
                        .to_le_bytes()
                        .into_iter()
                        .chain(i32::try_from(*x.numer()).unwrap().to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect(),
            10,
        )
    }

    pub fn float(code: u16, value: &Vec<f32>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            11,
        )
    }

    pub fn double(code: u16, value: &Vec<f64>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            12,
        )
    }

    pub fn ifd(code: u16, value: &Vec<u32>) -> Self {
        Tag::new(
            code,
            value.into_iter().map(|x| x.to_le_bytes()).flatten().collect(),
            13,
        )
    }

    pub fn unicode(code: u16, value: &str) -> Self {
        let mut bytes: Vec<u8> = value
            .encode_utf16()
            .map(|x| x.to_le_bytes())
            .flatten()
            .collect();
        bytes.push(0);
        Tag::new(code, bytes, 14)
    }

    pub fn complex(code: u16, value: &Vec<Complex<f32>>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| {
                    x.re.to_le_bytes()
                        .into_iter()
                        .chain(x.im.to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect(),
            15,
        )
    }

    pub fn long8(code: u16, value: &Vec<u64>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            16,
        )
    }

    pub fn slong8(code: u16, value: &Vec<i64>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            17,
        )
    }

    pub fn ifd8(code: u16, value: &Vec<u64>) -> Self {
        Tag::new(
            code,
            value
                .into_iter()
                .map(|x| x.to_le_bytes())
                .flatten()
                .collect(),
            18,
        )
    }

    pub fn short_long_or_long8(code: u16, value: &Vec<u64>) -> Self {
        let m = *value.iter().max().unwrap();
        if m < 65536 {
            Tag::short(code, &value.into_iter().map(|x| *x as u16).collect())
        } else if m < 4294967296 {
            Tag::long(code, &value.into_iter().map(|x| *x as u32).collect())
        } else {
            Tag::long8(code, value)
        }
    }

    pub fn count(&self) -> u64 {
        let c = match self.ttype {
            1 => self.bytes.len(),      // BYTE
            2 => self.bytes.len(),      // ASCII
            3 => self.bytes.len() / 2,  // SHORT
            4 => self.bytes.len() / 4,  // LONG
            5 => self.bytes.len() / 8,  // RATIONAL
            6 => self.bytes.len(),      // SBYTE
            7 => self.bytes.len(),      // UNDEFINED
            8 => self.bytes.len() / 2,  // SSHORT
            9 => self.bytes.len() / 4,  // SLONG
            10 => self.bytes.len() / 8, // SRATIONAL
            11 => self.bytes.len() / 4, // FLOAT
            12 => self.bytes.len() / 8, // DOUBLE
            13 => self.bytes.len() / 4, // IFD
            14 => self.bytes.len() / 2, // UNICODE
            15 => self.bytes.len() / 8, // COMPLEX
            16 => self.bytes.len() / 8, // LONG8
            17 => self.bytes.len() / 8, // SLONG8
            18 => self.bytes.len() / 8, // IFD8
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
                self.offset + (TAG_SIZE - OFFSET_SIZE) as u64,
            ))?;
            ijtifffile.file.write(&offset.to_le_bytes())?;
            if ijtifffile.file.stream_position()? % 2 == 1 {
                ijtifffile.file.write(&[0u8])?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CompressedFrame {
    bytes: Vec<Vec<u8>>,
    image_width: u32,
    image_length: u32,
    tile_size: usize,
    bits_per_sample: u16,
    sample_format: u16,
}

#[derive(Clone, Debug)]
struct Frame {
    offsets: Vec<u64>,
    bytecounts: Vec<u64>,
    image_width: u32,
    image_length: u32,
    bits_per_sample: u16,
    sample_format: u16,
    tile_width: u16,
    tile_length: u16,
}

impl Frame {
    fn new(
        offsets: Vec<u64>,
        bytecounts: Vec<u64>,
        image_width: u32,
        image_length: u32,
        bits_per_sample: u16,
        sample_format: u16,
        tile_width: u16,
        tile_length: u16,
    ) -> Self {
        Frame {
            offsets,
            bytecounts,
            image_width,
            image_length,
            bits_per_sample,
            sample_format,
            tile_width,
            tile_length,
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
            fn bytes(&self) -> Vec<u8> {
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

#[derive(Clone, Debug)]
pub enum Colors {
    None,
    Colors(Vec<Vec<u8>>),
    Colormap(Vec<Vec<u8>>),
}

#[derive(Debug)]
pub struct IJTiffFile {
    file: File,
    frames: HashMap<(usize, usize, usize), Frame>,
    hashes: HashMap<u64, u64>,
    threads: HashMap<(usize, usize, usize), JoinHandle<CompressedFrame>>,
    pub compression_level: i32,
    pub colors: Colors,
    pub comment: Option<String>,
    pub px_size: Option<f64>,
    pub delta_z: Option<f64>,
    pub time_interval: Option<f64>,
    pub extra_tags: HashMap<Option<(usize, usize, usize)>, Vec<Tag>>,
}

impl Drop for IJTiffFile {
    fn drop(&mut self) {
        if let Err(e) = self.close() {
            println!("Error closing IJTiffFile: {:?}", e);
        }
    }
}

impl IJTiffFile {
    pub fn new(path: &str) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path)?;
        file.write(b"II")?;
        file.write(&43u16.to_le_bytes())?;
        file.write(&8u16.to_le_bytes())?;
        file.write(&0u16.to_le_bytes())?;
        file.write(&OFFSET.to_le_bytes())?;
        Ok(IJTiffFile {
            file,
            frames: HashMap::new(),
            hashes: HashMap::new(),
            threads: HashMap::new(),
            compression_level: DEFAULT_COMPRESSION_LEVEL,
            colors: Colors::None,
            comment: None,
            px_size: None,
            delta_z: None,
            time_interval: None,
            extra_tags: HashMap::new(),
        })
    }

    pub fn set_compression_level(&mut self, compression_level: i32) {
        self.compression_level = compression_level.max(-7).min(22);
    }

    pub fn description(&self, c_size: usize, z_size: usize, t_size: usize) -> String {
        let mut desc: String = String::from("ImageJ=1.11a");
        if let Colors::None = self.colors {
            desc += &format!("\nimages={}", c_size);
            desc += &format!("\nslices={}", z_size);
            desc += &format!("\nframes={}", t_size);
        } else {
            desc += &format!("\nimages={}", c_size * z_size * t_size);
            desc += &format!("\nchannels={}", c_size);
            desc += &format!("\nslices={}", z_size);
            desc += &format!("\nframes={}", t_size);
        };
        if c_size == 1 {
            desc += "\nmode=grayscale";
        } else {
            desc += "\nmode=composite";
        }
        desc += "\nhyperstack=true\nloop=false\nunit=micron";
        if let Some(delta_z) = self.delta_z {
            desc += &format!("\nspacing={}", delta_z);
        }
        if let Some(timeinterval) = self.time_interval {
            desc += &format!("\ninterval={}", timeinterval);
        }
        if let Some(comment) = &self.comment {
            desc += &format!("\ncomment={}", comment);
        }
        desc
    }

    fn get_czt(
        &self,
        frame_number: usize,
        channel: u8,
        c_size: usize,
        z_size: usize,
    ) -> (usize, usize, usize) {
        if let Colors::None = self.colors {
            (
                channel as usize,
                frame_number % z_size,
                frame_number / z_size,
            )
        } else {
            (
                frame_number % c_size,
                frame_number / c_size % z_size,
                frame_number / c_size / z_size,
            )
        }
    }

    fn spp_and_n_frames(&self, c_size: usize, z_size: usize, t_size: usize) -> (u8, usize) {
        if let Colors::None = &self.colors {
            (c_size as u8, z_size * t_size)
        } else {
            (1, c_size * z_size * t_size)
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
        if self.hashes.contains_key(&hash)
            && self.hash_check(&bytes, *self.hashes.get(&hash).unwrap())?
        {
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

    pub fn save<T>(&mut self, frame: Array2<T>, c: usize, z: usize, t: usize) -> Result<()>
    where
        T: Bytes + Clone + Send + Sync + Zero + 'static,
    {
        self.compress_frame(frame.reversed_axes(), c, z, t)?;
        Ok(())
    }

    fn compress_frame<T>(&mut self, frame: Array2<T>, c: usize, z: usize, t: usize) -> Result<()>
    where
        T: Bytes + Clone + Zero + Send + 'static,
    {
        fn compress<T>(frame: Array2<T>, compression_level: i32) -> CompressedFrame
        where
            T: Bytes + Clone + Zero,
        {
            let image_width = frame.shape()[0] as u32;
            let image_length = frame.shape()[1] as u32;
            let tile_size = 2usize
                .pow(
                    ((image_width as f64 * image_length as f64 / 2f64).log2() / 2f64).round()
                        as u32,
                )
                .max(16)
                .min(1024);
            let tiles = IJTiffFile::tile(frame.reversed_axes(), tile_size);
            let byte_tiles: Vec<Vec<u8>> = tiles
                .into_iter()
                .map(|tile| tile.map(|x| x.bytes()).into_iter().flatten().collect())
                .collect();
            let bytes = if byte_tiles.len() > 4 {
                byte_tiles
                    .into_par_iter()
                    .map(|x| encode_all(x, compression_level).unwrap())
                    .collect::<Vec<_>>()
            } else {
                byte_tiles
                    .into_iter()
                    .map(|x| encode_all(x, compression_level).unwrap())
                    .collect::<Vec<_>>()
            };
            CompressedFrame {
                bytes,
                image_width,
                image_length,
                tile_size,
                bits_per_sample: T::BITS_PER_SAMPLE,
                sample_format: T::SAMPLE_FORMAT,
            }
        }
        loop {
            self.collect_threads(false)?;
            if self.threads.len() < 48 {
                break;
            }
            sleep(Duration::from_millis(100));
        }
        let compression_level = self.compression_level;
        self.threads.insert(
            (c, z, t),
            thread::spawn(move || compress(frame, compression_level)),
        );
        Ok(())
    }

    fn collect_threads(&mut self, block: bool) -> Result<()> {
        for (c, z, t) in self.threads.keys().cloned().collect::<Vec<_>>() {
            if block || self.threads[&(c, z, t)].is_finished() {
                if let Some(thread) = self.threads.remove(&(c, z, t)) {
                    self.write_frame(thread.join().unwrap(), c, z, t)?;
                }
            }
        }
        Ok(())
    }

    fn write_frame(&mut self, frame: CompressedFrame, c: usize, z: usize, t: usize) -> Result<()> {
        let mut offsets = Vec::new();
        let mut bytecounts = Vec::new();
        for tile in frame.bytes {
            bytecounts.push(tile.len() as u64);
            offsets.push(self.write(&tile)?);
        }
        let frame = Frame::new(
            offsets,
            bytecounts,
            frame.image_width,
            frame.image_length,
            frame.bits_per_sample,
            frame.sample_format,
            frame.tile_size as u16,
            frame.tile_size as u16,
        );
        self.frames.insert((c, z, t), frame);
        Ok(())
    }

    fn tile<T: Clone + Zero>(frame: Array2<T>, size: usize) -> Vec<Array2<T>> {
        let shape = frame.shape();
        let (n, m) = (shape[0] / size, shape[1] / size);
        let mut tiles = Vec::new();
        for i in 0..n {
            for j in 0..m {
                tiles.push(
                    frame
                        .slice(s![i * size..(i + 1) * size, j * size..(j + 1) * size])
                        .to_owned(),
                );
            }
            if shape[1] % size != 0 {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(s![.., ..shape[1] - m * size])
                    .assign(&frame.slice(s![i * size..(i + 1) * size, m * size..]));
                tiles.push(tile);
            }
        }
        if shape[0] % size != 0 {
            for j in 0..m {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(s![..shape[0] - n * size, ..])
                    .assign(&frame.slice(s![n * size.., j * size..(j + 1) * size]));
                tiles.push(tile);
            }
            if shape[1] % size != 0 {
                let mut tile = Array2::<T>::zeros((size, size));
                tile.slice_mut(s![..shape[0] - n * size, ..shape[1] - m * size])
                    .assign(&frame.slice(s![n * size.., m * size..]));
                tiles.push(tile);
            }
        }
        tiles
    }

    fn get_colormap(&self, colormap: &Vec<Vec<u8>>, bits_per_sample: u16) -> Vec<u16> {
        let mut r = Vec::new();
        let mut g = Vec::new();
        let mut b = Vec::new();
        let n = 2usize.pow(bits_per_sample as u32 - 8);
        for color in colormap {
            r.extend(vec![(color[0] as u16) * 257; n]);
            g.extend(vec![(color[1] as u16) * 257; n]);
            b.extend(vec![(color[2] as u16) * 257; n]);
        }
        r.extend(g);
        r.extend(b);
        r
    }

    fn get_color(&self, colors: &Vec<u8>, bits_per_sample: u16) -> Vec<u16> {
        let mut c = Vec::new();
        let n = 2usize.pow(bits_per_sample as u32 - 8);
        for color in colors {
            for i in 0..256 {
                c.extend(vec![i * (*color as u16) / 255 * 257; n])
            }
        }
        c
    }

    fn close(&mut self) -> Result<()> {
        self.collect_threads(true)?;
        let mut c_size = 1;
        let mut z_size = 1;
        let mut t_size = 1;
        for (c, z, t) in self.frames.keys() {
            c_size = c_size.max(c + 1);
            z_size = z_size.max(z + 1);
            t_size = t_size.max(t + 1);
        }

        let mut where_to_write_next_ifd_offset = OFFSET - OFFSET_SIZE as u64;
        let mut warn = Vec::new();
        let (samples_per_pixel, n_frames) = self.spp_and_n_frames(c_size, t_size, z_size);
        for frame_number in 0..n_frames {
            if let Some(frame) = self
                .frames
                .get(&self.get_czt(frame_number, 0, c_size, z_size))
            {
                let mut offsets = Vec::new();
                let mut bytecounts = Vec::new();
                let mut frame_count = 0;
                for channel in 0..samples_per_pixel {
                    if let Some(frame_n) =
                        self.frames
                            .get(&self.get_czt(frame_number, channel, c_size, z_size))
                    {
                        offsets.extend(frame_n.offsets.iter());
                        bytecounts.extend(frame_n.bytecounts.iter());
                        frame_count += 1;
                    } else {
                        warn.push((frame_number, channel));
                    }
                }
                let mut ifd = IFD::new();
                ifd.tags.insert(Tag::long(256, &vec![frame.image_width]));
                ifd.tags.insert(Tag::long(257, &vec![frame.image_length]));
                ifd.tags
                    .insert(Tag::short(258, &vec![frame.bits_per_sample; frame_count]));
                ifd.tags.insert(Tag::short(259, &vec![COMPRESSION]));
                ifd.tags
                    .insert(Tag::ascii(270, &self.description(c_size, z_size, t_size)));
                ifd.tags.insert(Tag::short(277, &vec![frame_count as u16]));
                ifd.tags.insert(Tag::ascii(305, "tiffwrite_rs"));
                ifd.tags.insert(Tag::short(322, &vec![frame.tile_width]));
                ifd.tags.insert(Tag::short(323, &vec![frame.tile_length]));
                ifd.tags.insert(Tag::short_long_or_long8(324, &offsets));
                ifd.tags.insert(Tag::short_long_or_long8(325, &bytecounts));
                if frame.sample_format > 1 {
                    ifd.tags.insert(Tag::short(339, &vec![frame.sample_format]));
                }
                if let Some(px_size) = self.px_size {
                    let r = vec![Rational32::from_f64(px_size).unwrap()];
                    ifd.tags.insert(Tag::rational(282, &r));
                    ifd.tags.insert(Tag::rational(283, &r));
                }
                if let Colors::Colormap(_) = &self.colors {
                    ifd.tags.insert(Tag::short(262, &vec![3]));
                } else if let Colors::None = self.colors {
                    ifd.tags.insert(Tag::short(262, &vec![1]));
                }
                if frame_number == 0 {
                    if let Colors::Colormap(colormap) = &self.colors {
                        ifd.tags.insert(Tag::short(
                            320,
                            &self.get_colormap(colormap, frame.bits_per_sample),
                        ));
                    }
                }
                if frame_number < c_size {
                    if let Colors::Colors(colors) = &self.colors {
                        ifd.tags.insert(Tag::short(
                            320,
                            &self.get_color(&colors[frame_number], frame.bits_per_sample),
                        ));
                        ifd.tags.insert(Tag::short(262, &vec![3]));
                    }
                }
                if let Colors::None = &self.colors {
                    if c_size > 1 {
                        ifd.tags.insert(Tag::short(284, &vec![2]));
                    }
                }
                for channel in 0..samples_per_pixel {
                    let czt = self.get_czt(frame_number, channel, c_size, z_size);
                    if let Some(extra_tags) = self.extra_tags.get(&Some(czt)) {
                        for tag in extra_tags {
                            ifd.tags.insert(tag.to_owned());
                        }
                    }
                }
                if let Some(extra_tags) = self.extra_tags.get(&None) {
                    for tag in extra_tags {
                        ifd.tags.insert(tag.to_owned());
                    }
                }
                if frame_number == 0 {
                    ifd.tags.insert(Tag::ascii(
                        306,
                        &format!("{}", Utc::now().format("%Y:%m:%d %H:%M:%S")),
                    ));
                }
                where_to_write_next_ifd_offset = ifd.write(self, where_to_write_next_ifd_offset)?;
            } else {
                warn.push((frame_number, 0));
            }
            if warn.len() > 0 {
                println!("The following frames were not added to the tif file");
                for (frame_number, channel) in &warn {
                    let (c, z, t) = self.get_czt(*frame_number, *channel, c_size, z_size);
                    println!("{c}, {z}, {t}")
                }
                println!("Either you forgot them, \
                         or an error occurred and the tif file was closed prematurely.")
            }
        }
        self.file
            .seek(SeekFrom::Start(where_to_write_next_ifd_offset))?;
        self.file.write(&0u64.to_le_bytes())?;
        Ok(())
    }
}
