

use anyhow::Result;
use kreuzberg_paddle_ocr::Ocr;
const  TEST_IMAGE : &str =  "handled_Image2176_1080_1920_80.jpg"; 

fn main() -> Result<()> {
    // Пути к вашим моделям (PP-OCRv5)
    let det_path = "models/ppocrv5/PP-OCRv5_server_det_infer.onnx";
    let rec_path = "models/ppocrv5/PP-OCRv5_server_rec_infer.onnx";
    let alphabet_path = "models/ppocrv5/alphabet.txt";

    // Инициализируем OCR
    let mut ocr = Ocr::new(det_path, rec_path, alphabet_path)?;

    // Распознаём изображение
    let image_path = "test.jpg";
    let results = ocr.recognize_from_path(image_path)?;

    if results.is_empty() {
        println!("Текст не найден.");
    } else {
        for (idx, item) in results.iter().enumerate() {
            println!(
                "Блок {}: \"{}\" (координаты: x={}, y={}, w={}, h={})",
                idx + 1,
                item.text,
                item.bbox.x,
                item.bbox.y,
                item.bbox.w,
                item.bbox.h
            );
        }
    }

    Ok(())
}