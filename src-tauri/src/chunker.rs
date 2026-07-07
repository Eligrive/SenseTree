pub struct Chunk {
    pub text: String,
    pub chunk_index: usize,
}

pub struct Chunker;

impl Chunker {
    /// Découpe un texte en morceaux de taille fixe avec un chevauchement (overlap)
    /// en essayant de ne pas couper les phrases au milieu.
    pub fn slice_text(content: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        if content.is_empty() {
            return chunks;
        }

        let chars: Vec<char> = content.chars().collect();
        let mut start = 0;
        let mut chunk_index = 0;

        while start < chars.len() {
            // On calcule la fin théorique du morceau
            let mut end = std::cmp::min(start + chunk_size, chars.len());

            // 🧠 Optimisation : Si on n'est pas à la fin du document, 
            // on recule jusqu'au dernier espace ou point pour ne pas couper un mot.
            if end < chars.len() {
                let mut backup = end;
                while backup > start && backup > end - 100 {
                    if chars[backup] == '.' || chars[backup] == '\n' || chars[backup] == ' ' {
                        end = backup + 1; // On coupe juste après le caractère de ponctuation/espace
                        break;
                    }
                    backup -= 1;
                }
            }

            // Extraction du texte du morceau
            let chunk_text: String = chars[start..end].iter().collect();
            let cleaned_text = chunk_text.trim().to_string();

            if !cleaned_text.is_empty() {
                chunks.push(Chunk {
                    text: cleaned_text,
                    chunk_index,
                });
                chunk_index += 1;
            }

            // On avance le curseur en soustrayant le chevauchement (overlap)
            if end >= chars.len() {
                break; // Fin du document
            }
            
            start = end - std::cmp::min(overlap, end - start);
        }

        chunks
    }
}