use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, PartialEq)]
pub enum FileType {
    Text,               // Code source, Markdown, TXT -> Extraction directe
    Document,           // PDF, Word -> Extracteur de doc
    Image,              // PNG, JPG -> Module de Vision
    RequiresAIRouting,  // 🤖 C'est ici que ton LLM interviendra !
    Ignored,            // Poubelle (node_modules, etc.)
}

pub struct Parser;

impl Parser {
    pub fn determine_file_type(path: &Path) -> FileType {

        // Fichiers-poubelle systèmes (macOS AppleDouble, index Windows…) : ignorés.
        let file_name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if file_name.starts_with("._")
            || file_name.eq_ignore_ascii_case(".DS_Store")
            || file_name.eq_ignore_ascii_case("Thumbs.db")
            || file_name.eq_ignore_ascii_case("desktop.ini")
        {
            return FileType::Ignored;
        }

        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.len() == 0 {
                return FileType::Ignored;
            }
        }
        // 1. Le Bouclier CPU : On bloque le bruit de masse instantanément
        let path_str = path.to_string_lossy().to_lowercase();
        if path_str.contains("node_modules")
            || path_str.contains(".venv")
            || path_str.contains("\\target\\")
            || path_str.contains(".git") {
            return FileType::Ignored;
        }

        // 1.5 Routage par EXTENSION pour les types bien connus — PRIORITAIRE, car plus
        // fiable que les magic bytes : `infer` classe parfois un .docx (qui est un ZIP)
        // comme simple archive, ce qui l'enverrait à tort dans le routage « binaire ».
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if let Some(ft) = Self::route_by_extension(&ext) {
            return ft;
        }

        // 2. Détection rapide via Magic Bytes avec la crate "infer"
        let kind = infer::get_from_path(path);
        
        match kind {
            Ok(Some(k)) => {
                let mime = k.mime_type();
                
                // Si c'est une image ou un document connu
                if mime.starts_with("image/") {
                    return FileType::Image;
                }
                if mime == "application/pdf" || mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" {
                    return FileType::Document;
                }
                
                // 🤖 Format complexe (Archive, Binaire, etc.), on demande à l'IA
                if mime.starts_with("application/") && mime != "application/json" {
                    return FileType::RequiresAIRouting;
                }
            },
            Ok(None) | Err(_) => {
                // Si le système ne reconnaît pas le format
                if Self::is_valid_utf8(path) {
                    return FileType::Text; // C'est un script ou un fichier texte custom
                } else {
                    return FileType::RequiresAIRouting; // 🤖 Binaire inconnu, on laisse l'IA juger
                }
            }
        }

        // Par sécurité
        FileType::RequiresAIRouting
    }

    /// Routage par extension (minuscule) pour les types bien connus. Fonction PURE,
    /// testable : `None` si l'extension n'est pas reconnue (on retombe alors sur les
    /// magic bytes). `.docx`/`.pdf` doivent être des Documents même si `infer` voit un ZIP.
    fn route_by_extension(ext: &str) -> Option<FileType> {
        match ext {
            // Documents réellement extractibles (voir worker::extract_text).
            "pdf" | "docx" => Some(FileType::Document),
            // Images raster gérées par la vision.
            "png" | "jpg" | "jpeg" | "gif" | "bmp" | "webp" => Some(FileType::Image),
            // Texte / code / données lisibles → extraction directe.
            "txt" | "md" | "markdown" | "rst" | "csv" | "tsv" | "log" | "json" | "toml"
            | "yaml" | "yml" | "xml" | "html" | "htm" | "css" | "scss" | "ini" | "cfg"
            | "conf" | "env" | "tex" | "bib" | "rs" | "py" | "js" | "jsx" | "ts" | "tsx"
            | "c" | "h" | "cpp" | "hpp" | "cc" | "cxx" | "java" | "kt" | "kts" | "go"
            | "rb" | "php" | "sh" | "bash" | "zsh" | "ps1" | "bat" | "sql" | "r" | "swift"
            | "scala" | "lua" | "pl" | "pm" | "vue" | "svelte" | "dart" | "clj" | "hs"
            | "ml" | "gradle" | "properties" => Some(FileType::Text),
            _ => None,
        }
    }

    // Fonction rapide pour vérifier si un fichier inconnu est du texte lisible
    fn is_valid_utf8(path: &Path) -> bool {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };

        // On lit seulement les 512 premiers octets (très rapide)
        let mut buffer = [0; 512];
        let bytes_read = file.read(&mut buffer).unwrap_or(0);
        
        if bytes_read == 0 {
            return false;
        }

        std::str::from_utf8(&buffer[..bytes_read]).is_ok()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docx_et_pdf_sont_des_documents() {
        // Régression : un .docx (qui est un ZIP) doit être un Document, pas du binaire.
        assert_eq!(Parser::route_by_extension("docx"), Some(FileType::Document));
        assert_eq!(Parser::route_by_extension("pdf"), Some(FileType::Document));
    }

    #[test]
    fn images_et_texte_par_extension() {
        assert_eq!(Parser::route_by_extension("png"), Some(FileType::Image));
        assert_eq!(Parser::route_by_extension("jpeg"), Some(FileType::Image));
        assert_eq!(Parser::route_by_extension("txt"), Some(FileType::Text));
        assert_eq!(Parser::route_by_extension("rs"), Some(FileType::Text));
        assert_eq!(Parser::route_by_extension("json"), Some(FileType::Text));
    }

    #[test]
    fn extension_inconnue_retombe_sur_les_magic_bytes() {
        assert_eq!(Parser::route_by_extension("xyz"), None);
        assert_eq!(Parser::route_by_extension(""), None);
        // xlsx/pptx ne sont PAS extractibles → laissés aux magic bytes (contexte).
        assert_eq!(Parser::route_by_extension("xlsx"), None);
    }
}
