use anyhow::Result;
use ndarray::{s, Array2};
use tiffwrite::IJTiffFile;


fn main() -> Result<()> {
    println!("Hello World!");
    let mut f = IJTiffFile::new("foo.tif", (2, 1, 1))?;
    let mut arr = Array2::<u16>::zeros((100, 100));
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            arr[[i, j]] = i as u16;
        }
    }
    f.save(arr.to_owned(), 0, 0, 0, None)?;

    let mut arr = Array2::<u16>::zeros((100, 100));
    arr.slice_mut(s![64.., ..64]).fill(1);
    arr.slice_mut(s![..64, 64..]).fill(2);
    arr.slice_mut(s![64.., 64..]).fill(3);
    f.save(arr.to_owned(), 1, 0,0, None)?;
    Ok(())
}