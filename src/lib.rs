#[cfg(feature = "python")]
mod py;

use anyhow::Result;
use chrono::Utc;
use flate2::write::ZlibEncoder;
use ndarray::{s, ArcArray2, AsArray, Ix2};
use num::{traits::ToBytes, Complex, FromPrimitive, Rational32};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Duration;
use std::{cmp::Ordering, collections::HashMap};
use std::{
    thread,
    thread::{available_parallelism, sleep, JoinHandle},
};
use zstd::zstd_safe::CompressionLevel;
use zstd::{stream::Encoder, DEFAULT_COMPRESSION_LEVEL};

const TAG_SIZE: usize = 20;
const OFFSET_SIZE: usize = 8;
const OFFSET: u64 = 16;

/// Compression: deflate or zstd
#[derive(Clone, Debug)]
pub enum Compression {
    Deflate,
    Zstd(CompressionLevel),
}

impl Compression {
    fn index(&self) -> u16 {
        match self {
            Compression::Deflate => 8,
            Compression::Zstd(_) => 50000,
        }
    }
}

/// Image File Directory
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
struct IFD {
    tags: HashSet<Tag>,
}

impl IFD {
    /// new IFD with empty set of tags
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
            ijtifffile.file.write_all(&[0])?;
        }
        let offset = ijtifffile.file.stream_position()?;
        ijtifffile
            .file
            .write_all(&(tags.len() as u64).to_le_bytes())?;

        for tag in tags.iter_mut() {
            tag.write_tag(ijtifffile)?;
        }
        let where_to_write_next_ifd_offset = ijtifffile.file.stream_position()?;
        ijtifffile.file.write_all(&[0; OFFSET_SIZE])?;
        for tag in tags.iter() {
            tag.write_data(ijtifffile)?;
        }
        ijtifffile
            .file
            .seek(SeekFrom::Start(where_to_write_offset))?;
        ijtifffile.file.write_all(&offset.to_le_bytes())?;
        Ok(where_to_write_next_ifd_offset)
    }
}

/// Tiff tag, use one of the constructors to get a tag of a specific type
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

    pub fn short(code: u16, value: &[u16]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            3,
        )
    }

    pub fn long(code: u16, value: &[u32]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            4,
        )
    }

    pub fn rational(code: u16, value: &[Rational32]) -> Self {
        Tag::new(
            code,
            value
                .iter()
                .flat_map(|x| {
                    u32::try_from(*x.denom())
                        .unwrap()
                        .to_le_bytes()
                        .into_iter()
                        .chain(u32::try_from(*x.numer()).unwrap().to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect(),
            5,
        )
    }

    pub fn sbyte(code: u16, value: &[i8]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            6,
        )
    }

    pub fn sshort(code: u16, value: &[i16]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            8,
        )
    }

    pub fn slong(code: u16, value: &[i32]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            9,
        )
    }

    pub fn srational(code: u16, value: &[Rational32]) -> Self {
        Tag::new(
            code,
            value
                .iter()
                .flat_map(|x| {
                    x.denom()
                        .to_le_bytes()
                        .into_iter()
                        .chain(x.numer().to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect(),
            10,
        )
    }

    pub fn float(code: u16, value: &[f32]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            11,
        )
    }

    pub fn double(code: u16, value: &[f64]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            12,
        )
    }

    pub fn ifd(code: u16, value: &[u32]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            13,
        )
    }

    pub fn unicode(code: u16, value: &str) -> Self {
        let mut bytes: Vec<u8> = value.encode_utf16().flat_map(|x| x.to_le_bytes()).collect();
        bytes.push(0);
        Tag::new(code, bytes, 14)
    }

    pub fn complex(code: u16, value: &[Complex<f32>]) -> Self {
        Tag::new(
            code,
            value
                .iter()
                .flat_map(|x| {
                    x.re.to_le_bytes()
                        .into_iter()
                        .chain(x.im.to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect(),
            15,
        )
    }

    pub fn long8(code: u16, value: &[u64]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            16,
        )
    }

    pub fn slong8(code: u16, value: &[i64]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            17,
        )
    }

    pub fn ifd8(code: u16, value: &[u64]) -> Self {
        Tag::new(
            code,
            value.iter().flat_map(|x| x.to_le_bytes()).collect(),
            18,
        )
    }

    pub fn short_long_or_long8(code: u16, value: &[u64]) -> Self {
        let m = *value.iter().max().unwrap();
        if m < 65536 {
            Tag::short(code, &value.iter().map(|x| *x as u16).collect::<Vec<_>>())
        } else if m < 4294967296 {
            Tag::long(code, &value.iter().map(|x| *x as u32).collect::<Vec<_>>())
        } else {
            Tag::long8(code, value)
        }
    }

    /// get the number of values in the tag
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
        ijtifffile.file.write_all(&self.code.to_le_bytes())?;
        ijtifffile.file.write_all(&self.ttype.to_le_bytes())?;
        ijtifffile.file.write_all(&self.count().to_le_bytes())?;
        if self.bytes.len() <= OFFSET_SIZE {
            ijtifffile.file.write_all(&self.bytes)?;
            ijtifffile
                .file
                .write_all(&vec![0; OFFSET_SIZE - self.bytes.len()])?;
        } else {
            ijtifffile.file.write_all(&[0; OFFSET_SIZE])?;
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
            ijtifffile.file.write_all(&offset.to_le_bytes())?;
            if ijtifffile.file.stream_position()? % 2 == 1 {
                ijtifffile.file.write_all(&[0])?;
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
    tile_width: usize,
    tile_length: usize,
    bits_per_sample: u16,
    sample_format: u16,
}

impl CompressedFrame {
    fn new<T>(frame: ArcArray2<T>, compression: Compression) -> CompressedFrame
    where
        T: Bytes + Send + Sync,
    {
        let shape = frame.shape();
        let tile_size = 2usize
            .pow(((shape[0] as f64 * shape[1] as f64 / 2f64).log2() / 2f64).round() as u32)
            .clamp(16, 1024);

        let tile_width = tile_size;
        let tile_length = tile_size;
        let n = shape[0] / tile_width;
        let m = shape[1] / tile_length;
        let mut slices = Vec::new();
        for i in 0..n {
            for j in 0..m {
                slices.push((
                    i * tile_width,
                    (i + 1) * tile_width,
                    j * tile_length,
                    (j + 1) * tile_length,
                ));
            }
            if shape[1] % tile_length != 0 {
                slices.push((
                    i * tile_width,
                    (i + 1) * tile_width,
                    m * tile_length,
                    shape[1],
                ));
            }
        }
        if shape[0] % tile_width != 0 {
            for j in 0..m {
                slices.push((
                    n * tile_width,
                    shape[0],
                    j * tile_length,
                    (j + 1) * tile_length,
                ));
            }
            if shape[1] % tile_length != 0 {
                slices.push((n * tile_width, shape[0], m * tile_length, shape[1]));
            }
        }

        let bytes: Vec<_> = match compression {
            Compression::Deflate => {
                if slices.len() > 4 {
                    slices
                        .into_par_iter()
                        .map(|slice| {
                            CompressedFrame::compress_tile_deflate(
                                frame.clone(),
                                slice,
                                tile_size,
                                tile_size,
                            )
                            .unwrap()
                        })
                        .collect()
                } else {
                    slices
                        .into_iter()
                        .map(|slice| {
                            CompressedFrame::compress_tile_deflate(
                                frame.clone(),
                                slice,
                                tile_size,
                                tile_size,
                            )
                            .unwrap()
                        })
                        .collect()
                }
            }

            Compression::Zstd(level) => {
                if slices.len() > 4 {
                    slices
                        .into_par_iter()
                        .map(|slice| {
                            CompressedFrame::compress_tile_zstd(
                                frame.clone(),
                                slice,
                                tile_size,
                                tile_size,
                                level,
                            )
                            .unwrap()
                        })
                        .collect()
                } else {
                    slices
                        .into_iter()
                        .map(|slice| {
                            CompressedFrame::compress_tile_zstd(
                                frame.clone(),
                                slice,
                                tile_size,
                                tile_size,
                                level,
                            )
                            .unwrap()
                        })
                        .collect()
                }
            }
        };

        CompressedFrame {
            bytes,
            image_width: shape[1] as u32,
            image_length: shape[0] as u32,
            tile_width,
            tile_length,
            bits_per_sample: T::BITS_PER_SAMPLE,
            sample_format: T::SAMPLE_FORMAT,
        }
    }

    fn encode<W, T>(
        mut encoder: W,
        frame: ArcArray2<T>,
        slice: (usize, usize, usize, usize),
        tile_width: usize,
        tile_length: usize,
    ) -> Result<W>
    where
        W: Write,
        T: Bytes,
    {
        let bytes_per_sample = (T::BITS_PER_SAMPLE / 8) as usize;
        let shape = (slice.1 - slice.0, slice.3 - slice.2);
        for i in 0..shape.0 {
            encoder.write_all(
                &frame
                    .slice(s![slice.0..slice.1, slice.2..slice.3])
                    .slice(s![i, ..])
                    .map(|x| x.bytes())
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            )?;
            encoder.write_all(&vec![0; bytes_per_sample * (tile_width - shape.1)])?;
        }
        encoder.write_all(&vec![
            0;
            bytes_per_sample * tile_width * (tile_length - shape.0)
        ])?;
        Ok(encoder)
    }

    fn compress_tile_deflate<T>(
        frame: ArcArray2<T>,
        slice: (usize, usize, usize, usize),
        tile_width: usize,
        tile_length: usize,
    ) -> Result<Vec<u8>>
    where
        T: Bytes,
    {
        let mut encoder = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder = CompressedFrame::encode(encoder, frame, slice, tile_width, tile_length)?;
        Ok(encoder.finish()?)
    }

    fn compress_tile_zstd<T>(
        frame: ArcArray2<T>,
        slice: (usize, usize, usize, usize),
        tile_width: usize,
        tile_length: usize,
        compression_level: i32,
    ) -> Result<Vec<u8>>
    where
        T: Bytes,
    {
        let mut dest = Vec::new();
        let mut encoder = Encoder::new(&mut dest, compression_level)?;
        let bytes_per_sample = (T::BITS_PER_SAMPLE / 8) as usize;
        encoder.include_contentsize(true)?;
        encoder.set_pledged_src_size(Some((bytes_per_sample * tile_width * tile_length) as u64))?;
        encoder.include_checksum(true)?;
        encoder = CompressedFrame::encode(encoder, frame, slice, tile_width, tile_length)?;
        encoder.finish()?;
        Ok(dest)
    }
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
    #[allow(clippy::too_many_arguments)]
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

/// trait to convert numbers to bytes
pub trait Bytes {
    const BITS_PER_SAMPLE: u16;
    /// 1: unsigned int, 2: signed int, 3: float
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

/// what colormap to save in the tiff;
#[derive(Clone, Debug)]
pub enum Colors {
    None,
    /// gradient from black to rgb color, 1 vec per channel
    Colors(Vec<Vec<u8>>),
    /// vec of rgb colors
    Colormap(Vec<Vec<u8>>),
}

/// save 2d arrays in a tif file compatible with Fiji/ImageJ
#[derive(Debug)]
pub struct IJTiffFile {
    file: File,
    frames: HashMap<(usize, usize, usize), Frame>,
    hashes: HashMap<u64, u64>,
    threads: HashMap<(usize, usize, usize), JoinHandle<CompressedFrame>>,
    /// zstd: -7 ..= 22
    pub compression: Compression,
    pub colors: Colors,
    pub comment: Option<String>,
    /// um per pixel
    pub px_size: Option<f64>,
    /// um per slice
    pub delta_z: Option<f64>,
    /// s per frame
    pub time_interval: Option<f64>,
    /// extra tags; per frame: key = Some((c, z, t)), global: key = None
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
    /// create new tifffile from path, use it's save() method to save frames
    /// the file is finalized when it goes out of scope
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(path)?;
        file.write_all(b"II")?;
        file.write_all(&43u16.to_le_bytes())?;
        file.write_all(&8u16.to_le_bytes())?;
        file.write_all(&0u16.to_le_bytes())?;
        file.write_all(&OFFSET.to_le_bytes())?;
        Ok(IJTiffFile {
            file,
            frames: HashMap::new(),
            hashes: HashMap::new(),
            threads: HashMap::new(),
            compression: Compression::Zstd(DEFAULT_COMPRESSION_LEVEL),
            colors: Colors::None,
            comment: None,
            px_size: None,
            delta_z: None,
            time_interval: None,
            extra_tags: HashMap::new(),
        })
    }

    /// set compression: zstd(level) or deflate
    pub fn set_compression(&mut self, compression: Compression) {
        self.compression = compression;
    }

    /// to be saved in description tag (270)
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
        let mut buffer = vec![0; bytes.len()];
        self.file.read_exact(&mut buffer)?;
        let same = bytes == &buffer;
        self.file.seek(SeekFrom::Start(current_offset))?;
        Ok(same)
    }

    fn write(&mut self, bytes: &Vec<u8>) -> Result<u64> {
        let hash = IJTiffFile::hash(&bytes);
        if self.hashes.contains_key(&hash)
            && self.hash_check(bytes, *self.hashes.get(&hash).unwrap())?
        {
            Ok(*self.hashes.get(&hash).unwrap())
        } else {
            if self.file.stream_position()? % 2 == 1 {
                self.file.write_all(&[0])?;
            }
            let offset = self.file.stream_position()?;
            self.hashes.insert(hash, offset);
            self.file.write_all(bytes)?;
            Ok(offset)
        }
    }

    /// save a 2d array to the tiff file at channel c, slice z, and time t
    pub fn save<'a, A, T>(&mut self, frame: A, c: usize, z: usize, t: usize) -> Result<()>
    where
        A: AsArray<'a, T, Ix2>,
        T: Bytes + Clone + Send + Sync + 'static,
    {
        let n_threads = usize::from(available_parallelism()?);
        loop {
            self.collect_threads(false)?;
            if self.threads.len() < n_threads {
                break;
            }
            sleep(Duration::from_millis(100));
        }
        let compression = self.compression.clone();
        let frame = frame.into().to_shared();
        self.threads.insert(
            (c, z, t),
            thread::spawn(move || CompressedFrame::new(frame, compression)),
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
            frame.tile_width as u16,
            frame.tile_length as u16,
        );
        self.frames.insert((c, z, t), frame);
        Ok(())
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
                ifd.tags.insert(Tag::long(256, &[frame.image_width]));
                ifd.tags.insert(Tag::long(257, &[frame.image_length]));
                ifd.tags
                    .insert(Tag::short(258, &vec![frame.bits_per_sample; frame_count]));
                ifd.tags
                    .insert(Tag::short(259, &[self.compression.index()]));
                ifd.tags
                    .insert(Tag::ascii(270, &self.description(c_size, z_size, t_size)));
                ifd.tags.insert(Tag::short(277, &[frame_count as u16]));
                ifd.tags.insert(Tag::ascii(305, "tiffwrite_rs"));
                ifd.tags.insert(Tag::short(322, &[frame.tile_width]));
                ifd.tags.insert(Tag::short(323, &[frame.tile_length]));
                ifd.tags.insert(Tag::short_long_or_long8(324, &offsets));
                ifd.tags.insert(Tag::short_long_or_long8(325, &bytecounts));
                if frame.sample_format > 1 {
                    ifd.tags.insert(Tag::short(339, &[frame.sample_format]));
                }
                if let Some(px_size) = self.px_size {
                    let r = [Rational32::from_f64(px_size).unwrap()];
                    ifd.tags.insert(Tag::rational(282, &r));
                    ifd.tags.insert(Tag::rational(283, &r));
                    ifd.tags.insert(Tag::short(296, &[1]));
                }
                if let Colors::Colormap(_) = &self.colors {
                    ifd.tags.insert(Tag::short(262, &[3]));
                } else if let Colors::None = self.colors {
                    ifd.tags.insert(Tag::short(262, &[1]));
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
                        ifd.tags.insert(Tag::short(262, &[3]));
                    }
                }
                if let Colors::None = &self.colors {
                    if c_size > 1 {
                        ifd.tags.insert(Tag::short(284, &[2]));
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
            if !warn.is_empty() {
                println!("The following frames were not added to the tif file:");
                for (frame_number, channel) in &warn {
                    let (c, z, t) = self.get_czt(*frame_number, *channel, c_size, z_size);
                    println!("c: {c}, z: {z}, t: {t}")
                }
                println!(
                    "Either you forgot them, \
                         or an error occurred and the tif file was closed prematurely."
                )
            }
        }
        self.file
            .seek(SeekFrom::Start(where_to_write_next_ifd_offset))?;
        self.file.write_all(&0u64.to_le_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    /// An example of generating julia fractals.
    fn julia_test() -> Result<()> {
        let imgx = 800;
        let imgy = 600;

        let scalex = 3.0 / imgx as f32;
        let scaley = 3.0 / imgy as f32;

        let mut im_r = Array2::<u8>::zeros((imgy, imgx));
        let mut im_g = Array2::<u8>::zeros((imgy, imgx));
        let mut im_b = Array2::<u8>::zeros((imgy, imgx));
        for x in 0..imgx {
            for y in 0..imgy {
                im_r[[y, x]] = (0.3 * x as f32) as u8;
                im_b[[y, x]] = (0.3 * y as f32) as u8;

                let cx = y as f32 * scalex - 1.5;
                let cy = x as f32 * scaley - 1.5;

                let c = Complex::new(-0.4, 0.6);
                let mut z = Complex::new(cx, cy);

                let mut i = 0;
                while i < 255 && z.norm() <= 2.0 {
                    z = z * z + c;
                    i += 1;
                }

                im_g[[y, x]] = i as u8;
            }
        }

        let mut f = IJTiffFile::new("julia.tif")?;
        f.save(&im_r, 0, 0, 0)?;
        f.save(&im_g, 1, 0, 0)?;
        f.save(&im_b, 2, 0, 0)?;

        Ok(())
    }
}
