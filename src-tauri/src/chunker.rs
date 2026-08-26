//! Découpage de texte **structure-aware** (façon « recursive character splitter »).
//!
//! Au lieu de couper aveuglément tous les N caractères, on découpe d'abord sur des
//! frontières sémantiques (paragraphes, puis phrases), puis on regroupe ces unités
//! gloutonnement jusqu'à la taille cible, avec un chevauchement (overlap) repris sur
//! une frontière de mot. Résultat : des chunks qui ne coupent ni au milieu d'une
//! phrase ni au milieu d'un mot → meilleure qualité de retrieval.

pub struct Chunk {
    pub text: String,
    pub chunk_index: usize,
}

pub struct Chunker;

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Séparation en phrases : coupe après `.`/`!`/`?` suivi d'un blanc (ou fin), ou après
/// un saut de ligne. Simple et déterministe (pas de modèle).
fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        cur.push(c);
        let sentence_end = matches!(c, '.' | '!' | '?')
            && chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
        if sentence_end || c == '\n' {
            let t = cur.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
            cur.clear();
        }
    }
    let t = cur.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
    out
}

/// Unités atomiques : paragraphes ; les paragraphes plus longs que `max` sont
/// redécoupés en phrases ; une phrase encore trop longue est coupée durement (fenêtres
/// de `max` caractères) — garantit que chaque unité tient dans la taille cible.
fn split_units(content: &str, max: usize) -> Vec<String> {
    let mut units = Vec::new();
    for para in content.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        if char_len(para) <= max {
            units.push(para.to_string());
            continue;
        }
        for sent in split_sentences(para) {
            if char_len(&sent) <= max {
                units.push(sent);
            } else {
                let chars: Vec<char> = sent.chars().collect();
                for window in chars.chunks(max) {
                    units.push(window.iter().collect());
                }
            }
        }
    }
    units
}

/// Fin de `s` (~`n` caractères) repartant après le premier espace pour ne pas couper
/// un mot — sert de chevauchement entre deux chunks consécutifs.
fn overlap_tail(s: &str, n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        return s.trim().to_string();
    }
    let slice: String = chars[chars.len() - n..].iter().collect();
    match slice.find(' ') {
        Some(pos) => slice[pos + 1..].trim().to_string(),
        None => slice.trim().to_string(),
    }
}

fn push_chunk(chunks: &mut Vec<Chunk>, idx: &mut usize, text: &str) {
    let t = text.trim();
    if t.is_empty() {
        return;
    }
    chunks.push(Chunk { text: t.to_string(), chunk_index: *idx });
    *idx += 1;
}

impl Chunker {
    /// Découpe `content` en chunks d'environ `chunk_size` caractères avec `overlap` de
    /// chevauchement, en respectant les frontières de paragraphe / phrase / mot.
    pub fn slice_text(content: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
        let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
        let content = normalized.trim();
        if content.is_empty() {
            return Vec::new();
        }
        let chunk_size = chunk_size.max(1);
        // L'overlap ne peut pas dépasser la moitié de la taille (sinon on n'avance plus).
        let overlap = overlap.min(chunk_size / 2);

        let units = split_units(content, chunk_size);

        let mut chunks = Vec::new();
        let mut idx = 0usize;
        let mut cur = String::new();
        for unit in units {
            let sep = if cur.is_empty() { 0 } else { 1 };
            if !cur.is_empty() && char_len(&cur) + sep + char_len(&unit) > chunk_size {
                push_chunk(&mut chunks, &mut idx, &cur);
                cur = overlap_tail(&cur, overlap); // continuité entre chunks
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&unit);
        }
        push_chunk(&mut chunks, &mut idx, &cur);
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::Chunker;

    #[test]
    fn vide_donne_aucun_chunk() {
        assert!(Chunker::slice_text("", 100, 20).is_empty());
        assert!(Chunker::slice_text("   \n\n  ", 100, 20).is_empty());
    }

    #[test]
    fn texte_court_reste_un_seul_chunk_intact() {
        let c = Chunker::slice_text("Bonjour le monde.", 100, 20);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].text, "Bonjour le monde.");
        assert_eq!(c[0].chunk_index, 0);
    }

    #[test]
    fn indexation_contigue_et_taille_bornee() {
        let para = "Phrase une. Phrase deux. Phrase trois. Phrase quatre. Phrase cinq.";
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = Chunker::slice_text(&text, 40, 10);
        assert!(chunks.len() > 1);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_index, i, "index non contigu");
            assert!(!c.text.is_empty());
            // Overlap toléré, mais pas de dérive démesurée.
            assert!(c.text.chars().count() <= 40 * 2, "chunk trop gros : {}", c.text);
        }
    }

    #[test]
    fn phrase_geante_coupee_sans_panic() {
        let giant = "a".repeat(1000);
        let chunks = Chunker::slice_text(&giant, 100, 20);
        assert!(chunks.len() >= 10);
        for c in &chunks {
            assert!(c.text.chars().count() <= 100 * 2);
        }
    }

    #[test]
    fn unicode_ne_panique_pas() {
        let text = "café résumé naïve œuf. ".repeat(50);
        let chunks = Chunker::slice_text(&text, 60, 15);
        assert!(!chunks.is_empty());
    }
}
