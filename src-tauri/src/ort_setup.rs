//! Approvisionnement d'ONNX Runtime en **chargement dynamique**.
//!
//! fastembed est compilé en `ort-load-dynamic` : la bibliothèque ORT n'est pas
//! liée au build, elle est chargée au runtime depuis `ORT_DYLIB_PATH`. On fournit
//! nous-mêmes cette lib, téléchargée au premier lancement (comme le modèle) :
//!   * CPU par défaut (marche partout) ;
//!   * GPU (CUDA) à la demande si la case est cochée ET qu'un GPU NVIDIA est présent.
//!
//! Un seul binaire universel : le choix GPU/CPU se fait sans recompiler.

use anyhow::{anyhow, Context, Result};
use std::fs::{self, File};
use std::path::{Path, PathBuf};

const ORT_VERSION: &str = "1.20.0"; // doit correspondre à ort-sys 2.0.0-rc.9
const CPU_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-1.20.0.zip";
const GPU_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-win-x64-gpu-1.20.0.zip";

/// Détecte au runtime la présence d'un GPU NVIDIA exploitable (driver installé).
pub fn gpu_present() -> bool {
    #[cfg(windows)]
    {
        // Le driver NVIDIA installe `nvcuda.dll` dans System32 ; sa présence est un
        // bon indicateur qu'un GPU CUDA est disponible.
        let sysroot = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".to_string());
        return Path::new(&sysroot).join("System32").join("nvcuda.dll").exists();
    }
    #[allow(unreachable_code)]
    false
}

/// Prépare ONNX Runtime et positionne `ORT_DYLIB_PATH`. Renvoie `true` si la
/// variante GPU a été retenue. Doit être appelé AVANT toute utilisation de
/// fastembed (donc avant le worker d'indexation).
pub fn ensure_ort(data_dir: &Path, use_gpu: bool) -> Result<bool> {
    let want_gpu = use_gpu && gpu_present();
    let (subdir, url, label) = if want_gpu {
        ("gpu", GPU_URL, "GPU")
    } else {
        ("cpu", CPU_URL, "CPU")
    };

    let dir = data_dir.join("onnxruntime").join(format!("{ORT_VERSION}-{subdir}"));
    let dll = dir.join("onnxruntime.dll");

    if !dll.exists() {
        tracing::info!(
            "ONNX Runtime {label} absent : téléchargement (premier lancement, cela peut prendre un moment)…"
        );
        fs::create_dir_all(&dir).with_context(|| format!("création de {}", dir.display()))?;
        if let Err(e) = download_and_extract(url, &dir) {
            // On nettoie un dossier partiel pour ne pas bloquer un futur essai.
            let _ = fs::remove_dir_all(&dir);
            return Err(e);
        }
        if !dll.exists() {
            return Err(anyhow!("onnxruntime.dll introuvable après extraction de {url}"));
        }
        tracing::info!("ONNX Runtime {label} prêt : {}", dll.display());
    }

    std::env::set_var("ORT_DYLIB_PATH", &dll);
    Ok(want_gpu)
}

/// Télécharge l'archive ORT et extrait toutes les DLL (à plat) dans `dir`.
/// Pour la variante GPU, cela inclut les providers CUDA à côté d'onnxruntime.dll.
fn download_and_extract(url: &str, dir: &Path) -> Result<()> {
    let tmp = dir.join("download.zip");

    // reqwest bloquant + streaming vers un fichier (les archives GPU font ~340 Mo).
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .context("client HTTP")?;
    let mut resp = client
        .get(url)
        .send()
        .context("téléchargement d'ONNX Runtime")?
        .error_for_status()
        .context("réponse de téléchargement invalide")?;
    {
        let mut out = File::create(&tmp).with_context(|| format!("création de {}", tmp.display()))?;
        std::io::copy(&mut resp, &mut out).context("écriture de l'archive")?;
    }

    // Extraction : on ne garde que les fichiers .dll (aplatis).
    let file = File::open(&tmp)?;
    let mut archive = zip::ZipArchive::new(file).context("archive ORT illisible")?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let name = entry.name().to_string();
        if !name.to_lowercase().ends_with(".dll") {
            continue;
        }
        let file_name = Path::new(&name)
            .file_name()
            .map(|n| n.to_os_string())
            .ok_or_else(|| anyhow!("nom d'entrée zip invalide: {name}"))?;
        let out_path: PathBuf = dir.join(file_name);
        let mut out = File::create(&out_path)
            .with_context(|| format!("extraction de {}", out_path.display()))?;
        std::io::copy(&mut entry, &mut out)?;
    }

    let _ = fs::remove_file(&tmp);
    Ok(())
}
