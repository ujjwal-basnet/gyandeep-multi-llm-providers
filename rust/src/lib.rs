use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[pyo3(signature = (text, chunk_size=500, overlap=100))]
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_ascii_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks: Vec<String> = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        chunks.push(words[i..end].join(" "));
        i += step;
    }
    chunks
}

#[pyfunction]
fn vectors_to_pg_literals(embeddings: Vec<Vec<f64>>) -> Vec<String> {
    embeddings
        .par_iter()
        .map(|emb| {
            let mut s = String::with_capacity(emb.len() * 10 + 2);
            s.push('[');
            for (i, v) in emb.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                use std::fmt::Write;
                write!(s, "{:.6}", v).unwrap();
            }
            s.push(']');
            s
        })
        .collect()
}

#[pyfunction]
fn truncate_texts(texts: Vec<String>, max_chars: usize) -> Vec<String> {
    texts
        .par_iter()
        .map(|t| {
            if t.len() <= max_chars {
                t.clone()
            } else {
                let mut end = max_chars;
                while !t.is_char_boundary(end) {
                    end -= 1;
                }
                t[..end].to_owned()
            }
        })
        .collect()
}

#[pymodule]
fn gyandeep_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(vectors_to_pg_literals, m)?)?;
    m.add_function(wrap_pyfunction!(truncate_texts, m)?)?;
    Ok(())
}
