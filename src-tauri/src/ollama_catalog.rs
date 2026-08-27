//! Catalogue LIVE de la bibliothèque officielle Ollama.
//!
//! Les leaderboards ([`crate::benchmarks`], [`crate::catalog`]) disent quels modèles
//! sont BONS, jamais lesquels sont *populaires*, *récents* ou simplement *installables
//! en une commande*. C'est le rôle de ce module, et c'est ce qui permet au catalogue
//! de ne jamais vieillir : aucun nom de modèle n'est codé en dur ici.
//!
//! Deux sources publiques, sans clé :
//!   * `https://ollama.com/library` — rendu SERVEUR, donc parsable : nom, description,
//!     capacités (vision / tools / thinking), tailles proposées, nombre de pulls et
//!     date de mise à jour ABSOLUE. Une seule requête renvoie toute la bibliothèque
//!     (~240 modèles) ; le tri n'est que du réordonnancement, fait ici localement.
//!   * `https://ollama.com/library/<modele>/tags` — la liste des tags d'un modèle,
//!     donc les variantes de QUANTIFICATION, chacune avec sa taille, son contexte et
//!     ses modalités. C'est ce qui permet de dire « le q4_K_M tient dans 8 Go, pas le
//!     q8_0 » plutôt que de se contenter du tag par défaut.
//!
//! Le registre OCI (`registry.ollama.ai`) donnerait la taille à l'octet près, mais il
//! faut l'interroger tag par tag : la page `/tags` livre la même information — à 0,05
//! Go près, sans effet sur un verdict VRAM — en une seule requête par modèle.
//!
//! Le parsing HTML est fragile par nature : il est donc entièrement TOLÉRANT (un champ
//! absent vaut `None`, jamais une erreur) et un parse vide déclenche le repli sur le
//! cache, exactement comme une panne réseau. L'utilisateur ne voit jamais un catalogue
//! vide à cause d'un changement de mise en page chez Ollama.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const LIBRARY_URL: &str = "https://ollama.com/library";
/// La bibliothèque bouge tous les jours (les leaderboards, eux, sont hebdomadaires).
const TTL_SECS: i64 = 24 * 3600;

/// Un modèle de la bibliothèque officielle Ollama.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OllamaModel {
    /// Nom d'installation, directement utilisable : `ollama pull <name>`.
    pub name: String,
    pub description: String,
    /// `vision`, `tools`, `thinking`, `embedding`, `audio`, `cloud`…
    pub capabilities: Vec<String>,
    /// Tailles proposées telles qu'affichées : `["4b", "9b", "27b"]`.
    pub sizes: Vec<String>,
    /// Téléchargements, normalisés pour le tri (`18.5M` → 18 500 000).
    pub pulls: u64,
    /// Le libellé d'origine, pour l'afficher sans réinventer le formatage.
    pub pulls_label: String,
    pub tags: Option<u32>,
    /// Date de mise à jour telle qu'annoncée (`Aug 26, 2026 11:07 PM UTC`).
    pub updated: Option<String>,
    /// La même, en jours depuis l'epoch : c'est elle qui sert au tri par récence.
    pub updated_day: Option<i64>,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    fetched_at: i64,
    entries: Vec<OllamaModel>,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64
}

fn cache_path(data_dir: &Path) -> PathBuf {
    data_dir.join("ollama-library.json")
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .user_agent("SenseTree")
        .timeout(Duration::from_secs(45))
        .build()?)
}

/// Bibliothèque Ollama complète. Cache 24 h ; en cas d'échec réseau OU de parse vide,
/// on sert le cache même périmé plutôt que de laisser l'utilisateur sans catalogue.
pub async fn library(data_dir: &Path, refresh: bool) -> Result<Vec<OllamaModel>> {
    let cached: Option<Cache> = std::fs::read_to_string(cache_path(data_dir))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());

    if !refresh {
        if let Some(c) = &cached {
            if now() - c.fetched_at < TTL_SECS {
                return Ok(c.entries.clone());
            }
        }
    }

    match fetch().await {
        Ok(entries) if !entries.is_empty() => {
            let c = Cache { fetched_at: now(), entries: entries.clone() };
            if let Ok(s) = serde_json::to_string(&c) {
                let _ = std::fs::create_dir_all(data_dir);
                let _ = std::fs::write(cache_path(data_dir), s);
            }
            Ok(entries)
        }
        result => match cached {
            Some(c) => {
                tracing::warn!("catalogue Ollama : source injoignable ou illisible, cache servi");
                Ok(c.entries)
            }
            None => match result {
                Ok(_) => Err(anyhow!(
                    "bibliothèque Ollama illisible (mise en page modifiée ?)"
                )),
                Err(e) => Err(e),
            },
        },
    }
}

async fn fetch() -> Result<Vec<OllamaModel>> {
    let html = client()?
        .get(LIBRARY_URL)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("lecture de la page bibliothèque Ollama")?;
    Ok(parse_library(&html))
}

/// Un tag précis d'un modèle : c'est ici que se joue le choix de la QUANTIFICATION
/// (`9b-q4_K_M` à 6,6 Go contre `9b-q8_0` à 11 Go — la frontière des 8 Go de VRAM).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OllamaTag {
    /// Suffixe seul (`9b-q4_K_M`), à coller derrière `modele:`.
    pub tag: String,
    /// Taille annoncée en octets. `None` pour les tags sans poids locaux (cloud).
    pub bytes: Option<u64>,
    /// Le libellé d'origine (`6.6GB`), pour l'afficher sans le reformater.
    pub size_label: Option<String>,
    /// Fenêtre de contexte annoncée (`256K`).
    pub context: Option<String>,
    /// Modalités d'entrée (`Text, Image`).
    pub modality: Option<String>,
}

/// Tags d'un modèle, avec leur taille — donc les variantes de quantification.
///
/// Une SEULE requête par modèle suffit : la page `/library/<modele>/tags` porte déjà
/// taille, contexte et modalités de chaque tag. C'est nettement moins coûteux que
/// d'interroger le registre OCI tag par tag, pour la même information.
pub async fn tags(data_dir: &Path, model: &str, refresh: bool) -> Result<Vec<OllamaTag>> {
    #[derive(Serialize, Deserialize, Default)]
    struct TagCache {
        entries: std::collections::HashMap<String, (i64, Vec<OllamaTag>)>,
    }

    let path = data_dir.join("ollama-tags.json");
    let mut cache: TagCache = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    if !refresh {
        if let Some((ts, v)) = cache.entries.get(model) {
            if now() - *ts < 7 * 24 * 3600 {
                return Ok(v.clone());
            }
        }
    }

    let html = client()?
        .get(format!("{LIBRARY_URL}/{model}/tags"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .with_context(|| format!("lecture des tags de {model}"))?;

    let parsed = parse_tags(&html, model);
    if parsed.is_empty() {
        // Même politique que la bibliothèque : un parse vide ne doit pas écraser un
        // cache valide ni remonter une liste vide comme si le modèle n'avait qu'un tag.
        if let Some((_, v)) = cache.entries.get(model) {
            tracing::warn!("tags de {model} illisibles, cache servi");
            return Ok(v.clone());
        }
        return Err(anyhow!("aucun tag lisible pour {model}"));
    }

    cache.entries.insert(model.to_string(), (now(), parsed.clone()));
    if let Ok(s) = serde_json::to_string(&cache) {
        let _ = std::fs::create_dir_all(data_dir);
        let _ = std::fs::write(&path, s);
    }
    Ok(parsed)
}

/// Tags de plusieurs modèles en parallèle. Un modèle illisible est simplement absent.
pub async fn tags_many(
    data_dir: &Path,
    models: Vec<String>,
) -> std::collections::HashMap<String, Vec<OllamaTag>> {
    let fetched =
        futures_util::future::join_all(models.iter().map(|m| tags(data_dir, m, false))).await;
    models
        .into_iter()
        .zip(fetched)
        .filter_map(|(m, r)| match r {
            Ok(v) => Some((m, v)),
            Err(e) => {
                tracing::debug!("tags non résolus ({e})");
                None
            }
        })
        .collect()
}

/// `6.6GB` → 6 600 000 000. Ollama affiche des unités DÉCIMALES (vérifié contre le
/// registre : 6 594 462 816 octets y sont annoncés « 6.6GB »).
fn parse_size(label: &str) -> Option<u64> {
    let l = label.trim().to_uppercase();
    let (num, mult) = if let Some(n) = l.strip_suffix("TB") {
        (n, 1e12)
    } else if let Some(n) = l.strip_suffix("GB") {
        (n, 1e9)
    } else if let Some(n) = l.strip_suffix("MB") {
        (n, 1e6)
    } else if let Some(n) = l.strip_suffix("KB") {
        (n, 1e3)
    } else {
        return None;
    };
    num.trim().parse::<f64>().ok().map(|v| (v * mult) as u64)
}

/// Extrait les tags de la page `/library/<modele>/tags`.
///
/// Chaque tag apparaît deux fois (variante mobile et variante bureau) : on ne garde
/// que la PREMIÈRE, qui porte déjà tout sur une ligne
/// (`<digest> · 6.6GB · 256K context window · Text, Image input · il y a 5 mois`).
pub fn parse_tags(html: &str, model: &str) -> Vec<OllamaTag> {
    let marker = format!("href=\"/library/{model}:");
    let mut out: Vec<OllamaTag> = Vec::new();
    let mut vus = std::collections::HashSet::new();

    for bloc in html.split(&marker).skip(1) {
        let Some(tag) = bloc.split('"').next() else { continue };
        if tag.is_empty() || !vus.insert(tag.to_string()) {
            continue;
        }
        // Fenêtre courte après le digest : au-delà, on mordrait sur le tag suivant.
        let fenetre = bloc
            .find("class=\"font-mono\"")
            .map(|k| &bloc[k..(k + 400).min(bloc.len())])
            .unwrap_or("");

        let size_label = mot_avant_unite(fenetre);
        let context = valeur_avant(fenetre, "context window");
        let modality = valeur_avant(fenetre, "input");

        out.push(OllamaTag {
            tag: tag.to_string(),
            bytes: size_label.as_deref().and_then(parse_size),
            size_label,
            context,
            modality,
        });
    }
    out
}

/// Repère `6.6GB` / `622MB` dans un fragment : un nombre immédiatement suivi d'une unité.
fn mot_avant_unite(s: &str) -> Option<String> {
    for unite in ["TB", "GB", "MB", "KB"] {
        let mut from = 0;
        while let Some(i) = s[from..].find(unite) {
            let pos = from + i;
            // On remonte le nombre qui précède (chiffres et point décimal).
            let debut = s[..pos]
                .rfind(|c: char| !c.is_ascii_digit() && c != '.')
                .map(|k| k + 1)
                .unwrap_or(0);
            let num = &s[debut..pos];
            if !num.is_empty() && num.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return Some(format!("{num}{unite}"));
            }
            from = pos + unite.len();
        }
    }
    None
}

/// Texte qui précède immédiatement un libellé (`256K` avant `context window`).
/// On s'arrête au séparateur `·` ou à toute balise, pour ne pas ramasser le voisin.
fn valeur_avant(s: &str, libelle: &str) -> Option<String> {
    let i = s.find(libelle)?;
    let avant = &s[..i];
    let debut = avant
        .rfind(|c: char| c == '·' || c == '>' || c == '\n')
        .map(|k| k + c_len(avant, k))
        .unwrap_or(0);
    let v = avant[debut..].trim();
    if v.is_empty() || v.len() > 40 {
        None
    } else {
        Some(v.to_string())
    }
}

/// Longueur en octets du caractère commençant à `i` (le séparateur `·` fait 2 octets).
fn c_len(s: &str, i: usize) -> usize {
    s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1)
}

// =============================================================================
// Parsing HTML — volontairement défensif : tout champ illisible vaut None.
// =============================================================================

/// Portion de `hay` située entre la première occurrence de `start` et le `end` suivant.
fn between<'a>(hay: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let i = hay.find(start)? + start.len();
    let rest = &hay[i..];
    let j = rest.find(end)?;
    Some(&rest[..j])
}

/// Toutes les portions entre `start` et `end`, sans chevauchement.
fn all_between<'a>(hay: &'a str, start: &str, end: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(i) = hay[from..].find(start) {
        let s = from + i + start.len();
        let Some(j) = hay[s..].find(end) else { break };
        out.push(&hay[s..s + j]);
        from = s + j + end.len();
    }
    out
}

/// Texte d'un badge : seulement ce qui suit le dernier `">`, donc après les classes CSS.
fn badge_text(fragment: &str) -> Option<&str> {
    let t = fragment.rsplit("\">").next()?.trim();
    if t.is_empty() || t.contains('<') {
        None
    } else {
        Some(t)
    }
}

/// Valeur du `<span>` qui PRÉCÈDE un libellé (`&nbsp;Pulls`, `&nbsp;Tags`).
///
/// La page met la valeur et son libellé dans deux spans frères ; remonter depuis le
/// libellé est plus robuste que de dépendre des espaces de l'attribut `<span >`.
fn label_value<'a>(card: &'a str, label: &str) -> Option<&'a str> {
    let i = card.find(label)?;
    let head = &card[..i];
    let close = head.rfind("</span>")?;
    let open = head[..close].rfind('>')?;
    let v = head[open + 1..close].trim();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// `18.5M` → 18 500 000 ; `2,503` → 2503. Sert uniquement au tri.
fn parse_pulls(s: &str) -> u64 {
    let s = s.trim().replace(',', "");
    let (num, mult) = match s.chars().last() {
        Some('K') | Some('k') => (&s[..s.len() - 1], 1_000f64),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1_000_000f64),
        Some('B') | Some('b') => (&s[..s.len() - 1], 1_000_000_000f64),
        _ => (s.as_str(), 1f64),
    };
    num.parse::<f64>().map(|v| (v * mult) as u64).unwrap_or(0)
}

fn month_num(m: &str) -> Option<i64> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    MONTHS.iter().position(|x| *x == m).map(|i| i as i64 + 1)
}

/// Jours depuis l'epoch (algorithme `days_from_civil` de Howard Hinnant).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `Aug 26, 2026 11:07 PM UTC` → jours depuis l'epoch. Seule la date compte pour le tri.
fn parse_date_day(s: &str) -> Option<i64> {
    let mut it = s.split_whitespace();
    let month = month_num(it.next()?)?;
    let day: i64 = it.next()?.trim_end_matches(',').parse().ok()?;
    let year: i64 = it.next()?.parse().ok()?;
    if !(1..=31).contains(&day) || !(2000..=2200).contains(&year) {
        return None;
    }
    Some(days_from_civil(year, month, day))
}

fn decode_entities(s: &str) -> String {
    s.replace("&#39;", "'")
        .replace("&quot;", "\"")
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
        .trim()
        .to_string()
}

/// Découpe la page en cartes `<li>` et en extrait un [`OllamaModel`] chacune.
pub fn parse_library(html: &str) -> Vec<OllamaModel> {
    let mut out = Vec::new();
    for card in html.split("<li ").skip(1) {
        let Some(name) = between(card, "href=\"/library/", "\"") else {
            continue;
        };
        if name.is_empty() || name.contains('/') {
            continue;
        }

        let description = between(card, "<p class=\"max-w-lg", "</p>")
            .and_then(|f| f.split_once('>'))
            .map(|(_, txt)| decode_entities(txt))
            .unwrap_or_default();

        // Capacités : badges indigo (vision/tools/thinking/embedding) et cyan (cloud).
        let mut capabilities: Vec<String> = all_between(card, "bg-indigo-50", "</span>")
            .into_iter()
            .chain(all_between(card, "bg-cyan-50", "</span>"))
            .filter_map(badge_text)
            .map(|s| s.to_string())
            .collect();
        capabilities.dedup();

        // Tailles : badges bleus (`4b`, `9b`, `300m`…).
        let sizes: Vec<String> = all_between(card, "bg-[#ddf4ff]", "</span>")
            .into_iter()
            .filter_map(badge_text)
            .filter(|s| {
                let l = s.to_lowercase();
                (l.ends_with('b') || l.ends_with('m'))
                    && l.chars().next().is_some_and(|c| c.is_ascii_digit())
            })
            .map(|s| s.to_lowercase())
            .collect();

        let pulls_label = label_value(card, "&nbsp;Pulls").unwrap_or("0").to_string();
        let tags = label_value(card, "&nbsp;Tags").and_then(|s| s.replace(',', "").parse().ok());

        // Le premier `title=` de la carte est le nom du modèle ; on retient le premier
        // qui se lit comme une date.
        let updated = all_between(card, "title=\"", "\"")
            .into_iter()
            .find(|t| parse_date_day(t).is_some())
            .map(|s| s.to_string());
        let updated_day = updated.as_deref().and_then(parse_date_day);

        out.push(OllamaModel {
            name: name.to_string(),
            description,
            capabilities,
            sizes,
            pulls: parse_pulls(&pulls_label),
            pulls_label,
            tags,
            updated,
            updated_day,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Une carte réelle, réduite : si Ollama change sa mise en page, ce test tombe
    /// AVANT que l'utilisateur ne voie un catalogue vide.
    const CARD: &str = r#"
    <li  class="flex items-baseline border-b border-neutral-200 py-6">
      <a href="/library/qwen3.5" class="group w-full space-y-5">
        <div  title="qwen3.5" class="flex flex-col">
          <h2 class="truncate text-xl font-medium"><div class="flex space-x-2 items-center">
            <span class="group-hover:underline truncate">qwen3.5</span></div></h2>
          <p class="max-w-lg break-words text-neutral-800 text-md">Qwen 3.5 is Alibaba&#39;s family.</p>
        </div>
        <div class="flex flex-col space-y-2">
          <div class="flex flex-wrap space-x-2">
            <span  class="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-600 sm:text-[13px]">vision</span>
            <span  class="inline-flex items-center rounded-md bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-600 sm:text-[13px]">tools</span>
            <span class="inline-flex items-center rounded-md bg-cyan-50 px-2 py-0.5 text-xs font-medium text-cyan-500 sm:text-[13px]">cloud</span>
            <span  class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2 py-0.5 text-xs font-medium text-blue-600 sm:text-[13px]">4b</span>
            <span  class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2 py-0.5 text-xs font-medium text-blue-600 sm:text-[13px]">9b</span>
          </div>
          <p class="my-4 flex space-x-5 text-[13px] font-medium text-neutral-500">
            <span class="flex items-center"><svg></svg>
              <span >18.5M</span><span class="hidden sm:flex">&nbsp;Pulls</span></span>
            <span class="flex items-center"><svg></svg>
              <span >64</span><span class="hidden sm:flex">&nbsp;Tags</span></span>
            <span class="flex items-center" title="May 21, 2026 7:08 PM UTC"><svg></svg>
              <span class="hidden sm:flex">Updated&nbsp;</span><span >3 months ago</span></span>
          </p>
        </div>
      </a>
    </li>"#;

    #[test]
    fn parse_une_carte_reelle() {
        let v = parse_library(CARD);
        assert_eq!(v.len(), 1, "une carte attendue");
        let m = &v[0];
        assert_eq!(m.name, "qwen3.5");
        assert_eq!(m.description, "Qwen 3.5 is Alibaba's family.");
        assert_eq!(m.capabilities, vec!["vision", "tools", "cloud"]);
        assert_eq!(m.sizes, vec!["4b", "9b"]);
        assert_eq!(m.pulls_label, "18.5M");
        assert_eq!(m.pulls, 18_500_000);
        assert_eq!(m.tags, Some(64));
        assert_eq!(m.updated.as_deref(), Some("May 21, 2026 7:08 PM UTC"));
        assert_eq!(m.updated_day, Some(days_from_civil(2026, 5, 21)));
    }

    #[test]
    fn pulls_normalises_pour_le_tri() {
        assert_eq!(parse_pulls("18.5M"), 18_500_000);
        assert_eq!(parse_pulls("83.6M"), 83_600_000);
        assert_eq!(parse_pulls("135.7K"), 135_700);
        assert_eq!(parse_pulls("2,503"), 2_503);
        assert_eq!(parse_pulls("520"), 520);
        // Illisible → 0, jamais une erreur : un modèle sans compteur reste listé.
        assert_eq!(parse_pulls("n/a"), 0);
    }

    #[test]
    fn dates_absolues_uniquement() {
        // Epoch, pour ancrer l'algorithme lui-même.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        // Le parseur, lui, refuse une année hors plage plausible : une date antérieure
        // à 2000 dans cette page signalerait un fragment mal découpé, pas un vieux modèle.
        assert_eq!(parse_date_day("Jan 1, 1970 12:00 AM UTC"), None);
        assert_eq!(
            parse_date_day("Jan 1, 2024 12:00 AM UTC"),
            Some(days_from_civil(2024, 1, 1))
        );
        // Ordre chronologique respecté.
        let a = parse_date_day("Aug 26, 2026 11:07 PM UTC").unwrap();
        let b = parse_date_day("May 21, 2026 7:08 PM UTC").unwrap();
        assert!(a > b);
        // « 3 months ago » n'est PAS une date : on refuse le relatif.
        assert_eq!(parse_date_day("3 months ago"), None);
        assert_eq!(parse_date_day("qwen3.5"), None);
    }

    /// Validation contre une VRAIE page, à lancer à la main quand on soupçonne
    /// qu'Ollama a changé sa mise en page :
    ///
    /// ```text
    /// curl -s https://ollama.com/library -o library.html
    /// OLLAMA_FIXTURE=library.html cargo test --lib parse_page_reelle -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "nécessite une page téléchargée (OLLAMA_FIXTURE)"]
    fn parse_page_reelle() {
        let path = std::env::var("OLLAMA_FIXTURE").expect("OLLAMA_FIXTURE non défini");
        let html = std::fs::read_to_string(&path).expect("fixture illisible");
        let models = parse_library(&html);

        println!("{} modèles analysés", models.len());
        assert!(models.len() > 50, "trop peu de modèles : mise en page changée ?");

        // Les champs qui portent la fonctionnalité doivent être là pour la quasi-totalité.
        let sans_desc = models.iter().filter(|m| m.description.is_empty()).count();
        let sans_pulls = models.iter().filter(|m| m.pulls == 0).count();
        let sans_date = models.iter().filter(|m| m.updated_day.is_none()).count();
        println!("sans description : {sans_desc} | sans pulls : {sans_pulls} | sans date : {sans_date}");
        assert!(sans_desc * 10 < models.len(), "descriptions majoritairement perdues");
        assert!(sans_pulls * 10 < models.len(), "compteurs de pulls majoritairement perdus");
        assert!(sans_date * 10 < models.len(), "dates majoritairement perdues");

        // Si le parsing dérape, on récupère du CSS plutôt qu'un mot-clé. On vérifie donc
        // la FORME (un mot court en minuscules) et non une liste fermée : Ollama ajoute
        // régulièrement des capacités (`audio` est arrivé après `vision`), et ce test ne
        // doit pas tomber pour cette raison-là.
        for m in &models {
            for c in &m.capabilities {
                assert!(
                    !c.is_empty()
                        && c.len() <= 16
                        && c.chars().all(|ch| ch.is_ascii_lowercase() || ch == '-'),
                    "capacité douteuse « {c} » sur {} : parsing déraillé ?",
                    m.name
                );
            }
        }

        // Les capacités dont dépendent les créneaux de SenseTree doivent être détectées.
        for attendu in ["vision", "tools", "thinking", "embedding"] {
            assert!(
                models.iter().any(|m| m.capabilities.iter().any(|c| c == attendu)),
                "aucun modèle ne porte « {attendu} » : badges non reconnus ?"
            );
        }

        let top = models.iter().max_by_key(|m| m.pulls).unwrap();
        println!("le plus populaire : {} ({} pulls)", top.name, top.pulls_label);
    }

    /// Un bloc de tag réel, dans sa variante compacte (celle qu'on parse).
    const TAG_BLOC: &str = r#"
      <a href="/library/qwen3.5:9b-q4_K_M" class="md:hidden flex flex-col group">
        <span class="group-hover:underline">qwen3.5:9b-q4_K_M</span>
        <div class="flex flex-col text-neutral-500 text-[13px]"><span>
          <span class="font-mono">6488c96fa5fa</span> · 6.6GB · 256K context window ·
          <span class="hidden sm:inline">Text, Image input · 5 months ago</span>
        </span></div>
      </a>
      <div class="hidden md:flex"><a href="/library/qwen3.5:9b-q4_K_M">qwen3.5:9b-q4_K_M</a></div>
      <a href="/library/qwen3.5:9b-q8_0" class="md:hidden">
        <div class="flex flex-col text-neutral-500 text-[13px]"><span>
          <span class="font-mono">aa11bb22cc33</span> · 11GB · 256K context window ·
          <span class="hidden sm:inline">Text, Image input · 5 months ago</span>
        </span></div>
      </a>"#;

    #[test]
    fn tags_portent_la_quantification_et_sa_taille() {
        let v = parse_tags(TAG_BLOC, "qwen3.5");
        // Chaque tag apparaît deux fois dans la page (mobile + bureau) : une seule entrée.
        assert_eq!(v.len(), 2, "dédoublonnage mobile/bureau");
        assert_eq!(v[0].tag, "9b-q4_K_M");
        assert_eq!(v[0].size_label.as_deref(), Some("6.6GB"));
        assert_eq!(v[0].bytes, Some(6_600_000_000));
        assert_eq!(v[0].context.as_deref(), Some("256K"));
        assert_eq!(v[0].modality.as_deref(), Some("Text, Image"));
        // Le q8_0 du MÊME modèle pèse presque le double : c'est tout l'intérêt du choix.
        assert_eq!(v[1].tag, "9b-q8_0");
        assert_eq!(v[1].bytes, Some(11_000_000_000));
        assert!(v[1].bytes > v[0].bytes);
    }

    #[test]
    fn tailles_en_unites_decimales() {
        // Vérifié contre le registre OCI : 6 594 462 816 octets sont annoncés « 6.6GB ».
        assert_eq!(parse_size("6.6GB"), Some(6_600_000_000));
        assert_eq!(parse_size("11GB"), Some(11_000_000_000));
        assert_eq!(parse_size("622MB"), Some(622_000_000));
        assert_eq!(parse_size("1.0GB"), Some(1_000_000_000));
        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("256K context"), None);
    }

    /// Comme pour la bibliothèque : à lancer à la main contre une vraie page.
    ///
    /// ```text
    /// curl -s https://ollama.com/library/qwen3.5/tags -o tags.html
    /// OLLAMA_TAGS_FIXTURE=tags.html cargo test --lib parse_tags_reels -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "nécessite une page téléchargée (OLLAMA_TAGS_FIXTURE)"]
    fn parse_tags_reels() {
        let path = std::env::var("OLLAMA_TAGS_FIXTURE").expect("OLLAMA_TAGS_FIXTURE non défini");
        let html = std::fs::read_to_string(&path).expect("fixture illisible");
        let v = parse_tags(&html, "qwen3.5");

        println!("{} tags", v.len());
        assert!(v.len() > 10, "trop peu de tags : mise en page changée ?");

        // Seuls les tags SANS poids locaux (cloud) peuvent ne pas avoir de taille.
        let sans_taille: Vec<&str> =
            v.iter().filter(|t| t.bytes.is_none()).map(|t| t.tag.as_str()).collect();
        println!("sans taille : {sans_taille:?}");
        assert!(sans_taille.len() * 5 < v.len(), "tailles majoritairement perdues");

        // Les variantes de quantification doivent être présentes et distinctes — à
        // NOMBRE DE PARAMÈTRES ÉGAL. Comparer le premier q4 au premier q8 du modèle
        // n'a aucun sens : les tags sont ordonnés par taille, donc `0.8b-q8_0` pèse
        // légitimement moins que `2b-q4_K_M`.
        let paires: Vec<(&OllamaTag, &OllamaTag)> = v
            .iter()
            .filter(|t| t.tag.ends_with("-q4_K_M"))
            .filter_map(|q4| {
                let base = q4.tag.trim_end_matches("-q4_K_M");
                v.iter()
                    .find(|t| t.tag == format!("{base}-q8_0"))
                    .map(|q8| (q4, q8))
            })
            .collect();
        assert!(!paires.is_empty(), "aucune paire q4_K_M / q8_0 comparable");
        for (q4, q8) in paires {
            println!("{} : {:?} | {} : {:?}", q4.tag, q4.size_label, q8.tag, q8.size_label);
            assert!(
                q8.bytes > q4.bytes,
                "{} devrait peser plus que {}",
                q8.tag,
                q4.tag
            );
        }

        for t in &v {
            assert!(!t.tag.contains('<'), "tag mal découpé : {}", t.tag);
        }
    }

    #[test]
    fn page_illisible_donne_zero_modele_pas_une_panne() {
        // Un parse vide déclenche le repli sur cache dans `library()` ; il ne doit
        // surtout pas paniquer.
        assert!(parse_library("<html><body>rien ici</body></html>").is_empty());
        assert!(parse_library("").is_empty());
    }
}
