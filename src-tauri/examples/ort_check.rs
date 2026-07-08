//! Vérification runtime du chemin CPU en chargement dynamique :
//! télécharge la lib ORT CPU, positionne ORT_DYLIB_PATH, puis charge fastembed
//! et embedde une phrase. `cargo run --example ort_check`

fn main() -> anyhow::Result<()> {
    let dir = std::env::temp_dir().join("sensetree_ort_check");
    std::fs::create_dir_all(&dir)?;

    let gpu = sensetree_lib::ort_setup::ensure_ort(&dir, false)?;
    println!(
        "ensure_ort OK (gpu={gpu}) ; ORT_DYLIB_PATH = {:?}",
        std::env::var("ORT_DYLIB_PATH")
    );

    let model = fastembed::TextEmbedding::try_new(
        fastembed::InitOptions::new(fastembed::EmbeddingModel::MultilingualE5Small)
            .with_show_download_progress(false),
    )?;
    let embeddings = model.embed(vec!["bonjour le monde".to_string()], None)?;
    println!(
        "OK : embedding calculé, dimension = {}",
        embeddings[0].len()
    );
    Ok(())
}
