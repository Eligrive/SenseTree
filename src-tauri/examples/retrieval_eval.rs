//! Banc de mesure de la recherche : indexe un jeu de données au format BEIR avec le
//! code de l'application (découpage, contextual retrieval, LanceDB, BM25, RRF,
//! reranking, dédoublonnage) et écrit un fichier de run TREC par configuration.
//!
//! ```text
//! cargo run --release --example retrieval_eval -- \
//!     --data ../research/data/scifact --out ../research/runs/pipeline/e5-small/scifact
//! ```
//!
//! Seule différence avec l'indexation réelle : le « sens » d'un document est son
//! extrait (repli de l'application quand la qualification LLM est désactivée).
//!
//! `--gpu true` calcule les embeddings du corpus et les logits du reranker sur CUDA.
//! Les latences publiées sont toujours mesurées sur CPU, par des instances dédiées,
//! sur les `--timing-queries` premières requêtes.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use sensetree_lib::chunker::Chunker;
use sensetree_lib::config::{Bm25Analyzer, EmbeddingConfig, EmbeddingMode};
use sensetree_lib::providers::{EmbeddingProvider, LocalEmbedder, Reranker};
use sensetree_lib::search::{
    candidate_pool, dense_candidates_only, finalize, fuse_rrf, logits_by_position, rerank_inputs,
    rerank_pool, Candidate, SearchResult,
};
use sensetree_lib::vectordb::{ChunkVector, SearchHit, VectorDb};
use sensetree_lib::worker::{chunk_texts, summary_of};

struct Args {
    data: PathBuf,
    out: PathBuf,
    model: String,
    reranker: String,
    analyzers: Vec<Bm25Analyzer>,
    rerank_analyzer: Bm25Analyzer,
    chunk_size: usize,
    overlap: usize,
    limit: usize,
    split: String,
    depths: Vec<usize>,
    timing_queries: usize,
    cache: PathBuf,
    gpu: bool,
    rerank: bool,
}

fn parse_analyzer(s: &str) -> Result<Bm25Analyzer> {
    serde_json::from_value(json!(s)).map_err(|_| anyhow!("analyseur inconnu : {s}"))
}

fn analyzer_name(a: Bm25Analyzer) -> String {
    serde_json::to_value(a).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()
}

fn parse_args() -> Result<Args> {
    let mut map: HashMap<String, String> = HashMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let key = k.trim_start_matches("--").to_string();
        let val = it.next().ok_or_else(|| anyhow!("valeur manquante pour {k}"))?;
        map.insert(key, val);
    }
    let get = |k: &str, d: &str| map.get(k).cloned().unwrap_or_else(|| d.to_string());
    let list = |k: &str, d: &str| -> Vec<String> {
        get(k, d).split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    };
    Ok(Args {
        data: PathBuf::from(map.get("data").ok_or_else(|| anyhow!("--data requis"))?),
        out: PathBuf::from(map.get("out").ok_or_else(|| anyhow!("--out requis"))?),
        model: get("model", "multilingual-e5-small"),
        reranker: get("reranker", "bge-reranker-v2-m3"),
        analyzers: list("analyzers", "english,french,neutral")
            .iter()
            .map(|s| parse_analyzer(s))
            .collect::<Result<_>>()?,
        rerank_analyzer: parse_analyzer(&get("rerank-analyzer", "english"))?,
        chunk_size: get("chunk-size", "1000").parse()?,
        overlap: get("overlap", "200").parse()?,
        limit: get("limit", "30").parse()?,
        split: get("split", "test"),
        depths: list("depths", "10,20,30,50,90").iter().map(|s| s.parse()).collect::<Result<_, _>>()?,
        timing_queries: get("timing-queries", "50").parse()?,
        cache: PathBuf::from(get("cache", "../research/cache")),
        gpu: get("gpu", "false") == "true",
        rerank: get("rerank", "true") == "true",
    })
}

#[derive(Deserialize)]
struct Doc {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct Query {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let f = std::fs::File::open(path).with_context(|| format!("ouverture de {}", path.display()))?;
    std::io::BufReader::new(f)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}

fn read_qrels(path: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let text = std::fs::read_to_string(path)?;
    for line in text.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() == 3 && cols[2].parse::<f32>().unwrap_or(0.0) > 0.0 {
            out.entry(cols[0].to_string()).or_default().push(cols[1].to_string());
        }
    }
    Ok(out)
}

/// Nom de fichier tiré du titre, comme le serait une note ou un document enregistré.
fn file_name_for(doc: &Doc) -> String {
    let base: String = doc
        .title
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' || c == '-' { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(80)
        .collect();
    if base.is_empty() {
        format!("{}.txt", doc.id)
    } else {
        format!("{base}.txt")
    }
}

fn doc_path(dataset: &str, doc: &Doc) -> String {
    format!("/bench/{dataset}/{}/{}", doc.id, file_name_for(doc))
}

fn doc_id_of(path: &str) -> String {
    path.split('/').nth(3).unwrap_or_default().to_string()
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' }).collect()
}

fn write_run(path: &Path, runs: &BTreeMap<String, Vec<SearchResult>>, tag: &str) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (qid, results) in runs {
        for (rank, r) in results.iter().enumerate() {
            let score = (results.len() - rank) as f32;
            writeln!(f, "{qid} Q0 {} {} {score:.1} {tag}", doc_id_of(&r.path), rank + 1)?;
        }
    }
    Ok(())
}

/// Logits du reranker déjà calculés, par (requête, chemin, chunk). Les candidats sont
/// largement les mêmes d'un modèle d'embedding à l'autre : le cache évite de repayer
/// le cross-encoder, qui domine le coût du banc.
struct LogitCache {
    path: PathBuf,
    map: HashMap<String, f32>,
    pending: Vec<(String, f32)>,
}

impl LogitCache {
    fn open(path: PathBuf) -> Result<Self> {
        let mut map = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                if let Some((k, v)) = line.rsplit_once('\t') {
                    if let Ok(v) = v.parse() {
                        map.insert(k.to_string(), v);
                    }
                }
            }
        }
        Ok(LogitCache { path, map, pending: Vec::new() })
    }

    fn key(qid: &str, c: &Candidate) -> String {
        format!("{qid}\t{}\t{}", c.path, c.chunk_index)
    }

    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        for (k, v) in self.pending.drain(..) {
            writeln!(f, "{k}\t{v}")?;
        }
        Ok(())
    }
}

#[derive(Default, Serialize)]
struct Timings {
    embed_query_ms: Vec<f64>,
    dense_ms: Vec<f64>,
    bm25_ms: Vec<f64>,
    fusion_ms: Vec<f64>,
    rerank_ms: BTreeMap<usize, Vec<f64>>,
    /// (paires, ms) de chaque appel de masse au reranker, sur le périphérique de masse.
    bulk_rerank: Vec<(usize, f64)>,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into()))
        .with_writer(std::io::stderr)
        .init();
    let args = parse_args()?;
    let dataset = args.data.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    std::fs::create_dir_all(&args.out)?;
    std::fs::create_dir_all(&args.cache)?;

    let on_gpu = sensetree_lib::ort_setup::ensure_ort(&args.cache.join("ort"), args.gpu)?;
    eprintln!("ONNX Runtime : {}", if on_gpu { "CUDA" } else { "CPU" });
    let model_cache = args.cache.join("models");

    let docs: Vec<Doc> = read_jsonl(&args.data.join("corpus.jsonl"))?;
    let queries: HashMap<String, String> = read_jsonl::<Query>(&args.data.join("queries.jsonl"))?
        .into_iter()
        .map(|q| (q.id, q.text))
        .collect();
    let qrels = read_qrels(&args.data.join("qrels").join(format!("{}.tsv", args.split)))?;
    eprintln!("{dataset} : {} documents, {} requêtes", docs.len(), qrels.len());

    let load_embedder = |use_gpu: bool| {
        let cfg = EmbeddingConfig {
            mode: EmbeddingMode::Local,
            model: args.model.clone(),
            use_gpu,
            ..Default::default()
        };
        let cache_dir = model_cache.clone();
        tokio::task::spawn_blocking(move || LocalEmbedder::load(&cfg, 32, cache_dir))
    };
    // Les requêtes passent toujours par une instance CPU, comme dans l'application.
    // L'instance GPU ne sert qu'à l'indexation et libère la VRAM avant le reranker.
    let embedder_q = load_embedder(false).await??;

    let model_tag = sanitize(&args.model);
    let lance_dir = args.cache.join("lancedb").join(format!("{dataset}-{model_tag}-c{}-o{}", args.chunk_size, args.overlap));
    let complete = lance_dir.join("complete.json");
    let vector = VectorDb::open(&lance_dir.to_string_lossy(), embedder_q.dimensions()).await?;

    let mut indexing = json!(null);
    if complete.exists() {
        indexing = serde_json::from_str(&std::fs::read_to_string(&complete)?)?;
        eprintln!("index existant réutilisé : {}", lance_dir.display());
    } else {
        vector.clear().await?;
        let embedder = load_embedder(on_gpu).await??;
        let started = Instant::now();
        let mut n_chunks = 0usize;
        let mut chars = 0usize;
        let mut embed_secs = 0.0f64;
        for (bi, block) in docs.chunks(512).enumerate() {
            let mut texts_embed: Vec<String> = Vec::new();
            let mut per_doc: Vec<(String, Vec<String>)> = Vec::new();
            for doc in block {
                let content = if doc.title.trim().is_empty() {
                    doc.text.trim().to_string()
                } else {
                    format!("{}\n\n{}", doc.title.trim(), doc.text.trim())
                };
                let raw: Vec<String> = Chunker::slice_text(&content, args.chunk_size, args.overlap)
                    .into_iter()
                    .map(|c| c.text)
                    .collect();
                if raw.is_empty() {
                    continue;
                }
                let (stored, embed) = chunk_texts(&file_name_for(doc), "txt", &summary_of(&content), raw);
                chars += embed.iter().map(|t| t.chars().count()).sum::<usize>();
                texts_embed.extend(embed);
                per_doc.push((doc_path(&dataset, doc), stored));
            }
            let t = Instant::now();
            let vectors = embedder.embed_documents(texts_embed).await?;
            embed_secs += t.elapsed().as_secs_f64();
            let mut vit = vectors.into_iter();
            let mut files = Vec::with_capacity(per_doc.len());
            for (path, stored) in per_doc {
                let chunks: Vec<ChunkVector> = stored
                    .into_iter()
                    .enumerate()
                    .map(|(i, text)| ChunkVector {
                        chunk_index: i as i32,
                        text,
                        vector: vit.next().expect("un vecteur par chunk"),
                    })
                    .collect();
                n_chunks += chunks.len();
                files.push((path, chunks));
            }
            vector.append_files(files).await?;
            eprintln!("  indexé {} / {} documents", ((bi + 1) * 512).min(docs.len()), docs.len());
        }
        indexing = json!({
            "documents": docs.len(),
            "chunks": n_chunks,
            "embedded_chars": chars,
            "embedding_seconds": embed_secs,
            "chunks_per_second": n_chunks as f64 / embed_secs.max(1e-9),
            "total_seconds": started.elapsed().as_secs_f64(),
        });
        std::fs::write(&complete, serde_json::to_string_pretty(&indexing)?)?;
    }

    let load_reranker = |use_gpu: bool| {
        let name = args.reranker.clone();
        let cache = model_cache.clone();
        tokio::task::spawn_blocking(move || Reranker::load(&name, cache, use_gpu))
    };
    let reranker = if args.rerank { Some(load_reranker(on_gpu).await??) } else { None };
    let reranker_cpu = if args.rerank && on_gpu { Some(load_reranker(false).await??) } else { None };
    let mut logit_cache = LogitCache::open(args.cache.join("rerank").join(format!("{dataset}-{}.tsv", sanitize(&args.reranker))))?;

    let pool = candidate_pool(args.limit);
    let depth_max = rerank_pool(args.limit);
    let mut depths: Vec<usize> = args.depths.iter().copied().filter(|d| *d <= depth_max).collect();
    if !depths.contains(&depth_max) {
        depths.push(depth_max);
    }

    let mut runs: BTreeMap<String, BTreeMap<String, Vec<SearchResult>>> = BTreeMap::new();
    let mut timings = Timings::default();
    let qids: Vec<String> = qrels.keys().cloned().collect();

    // Les requêtes BM25 de chaque analyseur, calculées d'un bloc : changer d'analyseur
    // reconstruit l'index plein-texte.
    let mut sparse_by: BTreeMap<String, HashMap<String, Vec<SearchHit>>> = BTreeMap::new();
    for &analyzer in &args.analyzers {
        vector.set_bm25_analyzer(analyzer);
        vector.keyword_search("warmup", 1, None).await?;
        let mut per_q = HashMap::new();
        for qid in &qids {
            let t = Instant::now();
            let hits = vector.keyword_search(&queries[qid], pool, None).await.unwrap_or_default();
            if analyzer == args.rerank_analyzer && timings.bm25_ms.len() < args.timing_queries {
                timings.bm25_ms.push(ms(t));
            }
            per_q.insert(qid.clone(), hits);
        }
        sparse_by.insert(analyzer_name(analyzer), per_q);
    }
    let rerank_an = analyzer_name(args.rerank_analyzer);

    for (n, qid) in qids.iter().enumerate() {
        let query = queries.get(qid).ok_or_else(|| anyhow!("requête {qid} absente"))?.trim().to_string();

        let timed = n < args.timing_queries;
        let t = Instant::now();
        let qvec = embedder_q.embed_query(query.clone()).await?;
        if timed {
            timings.embed_query_ms.push(ms(t));
        }
        let t = Instant::now();
        let dense = vector.search(qvec, pool, None).await?;
        if timed {
            timings.dense_ms.push(ms(t));
        }

        let keep = |_: &str| true;
        runs.entry("dense".into()).or_default().insert(
            qid.clone(),
            finalize(dense_candidates_only(dense.clone()), None, args.limit, keep),
        );
        for (an, per_q) in &sparse_by {
            let sparse = per_q[qid].clone();
            runs.entry(format!("bm25-{an}")).or_default().insert(
                qid.clone(),
                finalize(fuse_rrf(Vec::new(), sparse.clone()), None, args.limit, keep),
            );
            let t = Instant::now();
            let fused = fuse_rrf(dense.clone(), sparse);
            if *an == rerank_an && timed {
                timings.fusion_ms.push(ms(t));
            }
            runs.entry(format!("hybrid-{an}")).or_default().insert(
                qid.clone(),
                finalize(fused, None, args.limit, keep),
            );
        }

        let Some(reranker) = reranker.as_ref() else {
            continue;
        };
        let cands = fuse_rrf(dense.clone(), sparse_by[&rerank_an][qid].clone());
        let top = cands.len().min(depth_max);
        let missing: Vec<usize> = (0..top)
            .filter(|&i| !logit_cache.map.contains_key(&LogitCache::key(qid, &cands[i])))
            .collect();
        if !missing.is_empty() || timed {
            let inputs = rerank_inputs(&cands, top);
            if timed {
                for &d in &depths {
                    let sub: Vec<String> = inputs[..d.min(top)].to_vec();
                    let t = Instant::now();
                    reranker_cpu.as_ref().unwrap_or(reranker).rerank(query.clone(), sub).await?;
                    timings.rerank_ms.entry(d).or_default().push(ms(t));
                }
            }
            if !missing.is_empty() {
                let sub: Vec<String> = missing.iter().map(|&i| inputs[i].clone()).collect();
                let pairs = sub.len();
                let t = Instant::now();
                // Lots de 16 paires : un lot unique de 90 passages longs dépasse les 8 Go
                // d'une RTX 2080 Super. Chaque paire est notée indépendamment, le
                // découpage ne change pas les scores.
                let mut order = Vec::with_capacity(pairs);
                for (bi, lot) in sub.chunks(16).enumerate() {
                    let part = reranker.rerank(query.clone(), lot.to_vec()).await?;
                    order.extend(part.into_iter().map(|(i, l)| (i + bi * 16, l)));
                }
                timings.bulk_rerank.push((pairs, ms(t)));
                if n < 3 {
                    eprintln!("  reranking : {pairs} paires en {:.0} ms", ms(t));
                }
                for (j, logit) in order {
                    let k = LogitCache::key(qid, &cands[missing[j]]);
                    logit_cache.map.insert(k.clone(), logit);
                    logit_cache.pending.push((k, logit));
                }
            }
        }
        let logits: Vec<f32> = (0..top).map(|i| logit_cache.map[&LogitCache::key(qid, &cands[i])]).collect();
        for &d in &depths {
            let order: Vec<(usize, f32)> = logits[..d.min(top)].iter().copied().enumerate().collect();
            let name = if d == depth_max {
                format!("hybrid-{rerank_an}-rerank")
            } else {
                format!("hybrid-{rerank_an}-rerank{d}")
            };
            runs.entry(name).or_default().insert(
                qid.clone(),
                finalize(cands.clone(), Some(logits_by_position(order, d.min(top))), args.limit, keep),
            );
        }

        if n % 25 == 24 {
            logit_cache.flush()?;
            eprintln!("  {} / {} requêtes", n + 1, qids.len());
        }
    }
    logit_cache.flush()?;

    for (system, run) in &runs {
        write_run(&args.out.join(format!("{system}.trec")), run, system)?;
    }
    let meta = json!({
        "dataset": dataset,
        "split": args.split,
        "model": args.model,
        "reranker": args.reranker,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "limit": args.limit,
        "candidate_pool": pool,
        "rerank_pool": depth_max,
        "rerank_analyzer": rerank_an,
        "queries": qids.len(),
        "indexing": indexing,
        "timings": timings,
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "bulk_device": if on_gpu { "cuda" } else { "cpu" },
        "timing_device": "cpu",
    });
    std::fs::write(args.out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    eprintln!("runs écrits dans {}", args.out.display());
    Ok(())
}
