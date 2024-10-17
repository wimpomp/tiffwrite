use anyhow::Result;
use ndarray::{s, Array2};
use tiffwrite::IJTiffFile;

fn main() -> Result<()> {
    println!("Hello World!");
    let mut f = IJTiffFile::new("foo.tif")?;
    f.compression_level = 10;
    let mut arr = Array2::<u16>::zeros((100, 100));
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            arr[[i, j]] = i as u16;
        }
    }
    f.save(arr.view(), 0, 0, 0)?;

    let mut arr = Array2::<u16>::zeros((100, 100));
    arr.slice_mut(s![64.., ..64]).fill(1);
    arr.slice_mut(s![..64, 64..]).fill(2);
    arr.slice_mut(s![64.., 64..]).fill(3);
    f.save(&arr, 1, 0, 0)?;
    Ok(())
}
