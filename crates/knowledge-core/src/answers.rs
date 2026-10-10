//! Resolve supplied passage references and scope absence-of-proof statements.
pub fn grounded_answer(raw: &str, evidence_count: usize) -> String {
    let mut output=String::new();let mut rest=raw;
    while let Some(open)=rest.find('[') {
        output.push_str(&rest[..open]);rest=&rest[open..];
        let Some(close)=rest.find(']') else {break;};
        let label=&rest[1..close];
        if label.starts_with('E') {
            let ids:Vec<_>=label.split(',').map(str::trim).collect();
            for (i,id) in ids.iter().enumerate() {
                if i>0 { output.push(' '); }
                match id.strip_prefix('E').and_then(|n|n.parse::<usize>().ok()).filter(|n|*n>0&&*n<=evidence_count) {
                    Some(n)=>output.push_str(&format!("[E{n}]")),
                    None=>output.push_str("(unavailable evidence reference)"),
                }
            }
        } else {output.push_str(&rest[..=close]);}
        rest=&rest[close+1..];
    }
    output.push_str(rest);
    // Retrieval cannot establish absence of evidence in the full video or
    // outside the archive. Keep that scope explicit in generated conclusions.
    for (phrase,replacement) in [
        ("the speaker does not provide independent verification","the supplied excerpts do not provide independent verification"),
        ("the source does not provide independent verification","the supplied excerpts do not provide independent verification"),
        ("the video does not provide independent verification","the supplied excerpts do not provide independent verification"),
        ("not confirmed by evidence","not independently confirmed by the retrieved excerpts"),
        ("not a confirmed fact","not independently confirmed by these retrieved excerpts"),
        ("not an established fact","not independently established by these retrieved excerpts"),
        ("no supporting documentation or verification provided","no independently verified documentation in these retrieved excerpts"),
        ("the speaker does not mention","in the retrieved excerpts, the speaker does not mention"),
        ("the speaker does not discuss","in the retrieved excerpts, the speaker does not discuss"),
    ] {
        let mut from=0;
        while let Some(index)=output[from..].to_ascii_lowercase().find(phrase) {
            let start=from+index;let replacement=if output.as_bytes()[start].is_ascii_uppercase(){let mut r=replacement.to_string();r.get_mut(..1).unwrap().make_ascii_uppercase();r}else{replacement.to_string()};output.replace_range(start..start+phrase.len(),&replacement);from=start+replacement.len();
        }
    }
    output
}
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}
/// Edit distance small enough to be a recognition or spelling variant:
/// "froin" for "from", "dont" for "don't". Different words such as "goyim"
/// and "globe" are not variants.
fn variant(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let allowed = match a.chars().count().min(b.chars().count()) {
        0..=2 => return false,
        3 => 1,
        _ => 2,
    };
    if a.chars().next() != b.chars().next() {
        return false;
    }
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let current = row[j + 1];
            row[j + 1] = (current + 1).min(row[j] + 1).min(previous + usize::from(x != *y));
            previous = current;
        }
    }
    row[b.len()] <= allowed
}
/// Whether every word of `quote` appears in order in one passage within a
/// window slightly longer than the quote: the quote may skip a few source
/// words but may not add or change any. Words must match exactly, except one
/// spelling variant per eight quoted words, so a corrected OCR typo passes but
/// "Account Name: Baal" does not match "account is named Baal".
fn quoted_in(quote: &[String], passages: &[Vec<String>]) -> bool {
    let quote: Vec<&String> = quote.iter().filter(|w| w.chars().count() > 1).collect();
    if quote.is_empty() {
        return true;
    }
    let slack = if quote.len() < 5 { 1 } else { (quote.len() / 5).max(2) };
    passages.iter().any(|p| {
        let width = (quote.len() + slack).min(p.len());
        (0..=p.len().saturating_sub(width)).any(|start| {
            let window = &p[start..start + width];
            let mut position = 0;
            let mut variants = quote.len() / 8;
            quote.iter().all(|q| {
                if let Some(offset) = window[position..].iter().position(|w| w == *q) {
                    position += offset + 1;
                    return true;
                }
                if variants > 0 {
                    if let Some(offset) = window[position..].iter().position(|w| variant(q, w)) {
                        variants -= 1;
                        position += offset + 1;
                        return true;
                    }
                }
                false
            })
        })
    })
}
/// Flag quotations of three or more words that the supplied passages do not
/// contain, so a reformatted or invented quote never reads as the source's words.
pub fn mark_unverified_quotes(answer: &str, passages: &[String]) -> String {
    mark_quotes(answer, passages, &[])
}
/// Like [`mark_unverified_quotes`], with `descriptions` holding automatic frame
/// descriptions: a small vision model's transcription of on-screen text is
/// often wrong, so a quote found only there is labeled as such.
pub fn mark_quotes(answer: &str, passages: &[String], descriptions: &[String]) -> String {
    let passages: Vec<Vec<String>> = passages.iter().map(|p| tokens(p)).collect();
    let descriptions: Vec<Vec<String>> = descriptions.iter().map(|p| tokens(p)).collect();
    let mut output = String::new();
    let mut rest = answer;
    loop {
        let Some((open, opener)) = rest.char_indices().find(|(_, c)| *c == '"' || *c == '\u{201c}') else { break };
        let closer = if opener == '"' { '"' } else { '\u{201d}' };
        let body_start = open + opener.len_utf8();
        let Some(length) = rest[body_start..].find(closer) else { break };
        let close = body_start + length + closer.len_utf8();
        output.push_str(&rest[..close]);
        let quote = tokens(&rest[body_start..body_start + length]);
        if quote.len() >= 3 && !quoted_in(&quote, &passages) {
            output.push_str(if quoted_in(&quote, &descriptions) {
                " (from an automatic frame description; wording may be invented)"
            } else {
                " (not verbatim in the cited passages)"
            });
        }
        rest = &rest[close..];
    }
    output.push_str(rest);
    output
}

/// Splits a streamed answer into finished sentences so speech can start before
/// generation ends. Each sentence receives the same citation and scope rewrites
/// as the final answer. Template labels are removed, and anything after the
/// paragraph that follows a `Scope:` label is dropped, matching the final text.
pub struct SentenceStream { buffer: String, evidence_count: usize, passages: Vec<String>, descriptions: Vec<String>, scoped: bool, closed: bool }
impl SentenceStream {
    pub fn new(evidence_count: usize) -> Self {
        Self { buffer: String::new(), evidence_count, passages: Vec::new(), descriptions: Vec::new(), scoped: false, closed: false }
    }
    /// Check quotations in each sentence against these passages.
    pub fn with_passages(mut self, passages: Vec<String>) -> Self {
        self.passages = passages;
        self
    }
    /// Automatic frame descriptions, checked as in [`mark_quotes`].
    pub fn with_descriptions(mut self, descriptions: Vec<String>) -> Self {
        self.descriptions = descriptions;
        self
    }
    pub fn push(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if self.closed { return out; }
        self.buffer.push_str(text);
        let mut from = 0;
        while let Some((end, newline)) = boundary(&self.buffer, from) {
            if self.scoped && self.buffer[..end].trim().is_empty() && self.buffer[end..].starts_with('\n') {
                self.closed = true; self.buffer.clear(); return out;
            }
            // Very short fragments such as "Yes." read better joined to the next sentence.
            if !newline && self.buffer[..end].trim().chars().count() < 24 { from = end; continue; }
            let piece: String = self.buffer.drain(..end).collect();
            from = 0;
            if let Some(sentence) = self.clean(&piece) { out.push(sentence); }
        }
        out
    }
    pub fn finish(&mut self) -> Option<String> {
        if self.closed { return None; }
        self.closed = true;
        let rest = std::mem::take(&mut self.buffer);
        self.clean(&rest)
    }
    fn clean(&mut self, piece: &str) -> Option<String> {
        let mut text = piece.trim();
        if let Some(rest) = text.strip_prefix("Answer:") { text = rest.trim_start(); }
        if let Some(rest) = text.strip_prefix("Scope:") { text = rest.trim_start(); self.scoped = true; }
        (!text.is_empty()).then(|| mark_quotes(&grounded_answer(text, self.evidence_count), &self.passages, &self.descriptions))
    }
}
/// Byte offset just past the next sentence end at or after `from`, and whether it was a line break.
fn boundary(text: &str, from: usize) -> Option<(usize, bool)> {
    let mut chars = text[from..].char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if c == '\n' { return Some((from + i + 1, true)); }
        if matches!(c, '.' | '!' | '?') && chars.peek().is_some_and(|(_, next)| next.is_whitespace()) {
            return Some((from + i + 1, false));
        }
    }
    None
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_actual_references_are_clickable_including_grouped_citations() {
        assert_eq!(grounded_answer("A [E1, E2], B [E0, E99], [note]",2),
            "A [E1] [E2], B (unavailable evidence reference) (unavailable evidence reference), [note]");
    }
    #[test]
    fn absence_of_proof_is_scoped_to_retrieval() {
        let answer=grounded_answer("The speaker does not provide independent verification; not confirmed by evidence [E1].",1);
        assert!(answer.contains("The supplied excerpts do not provide"));
        assert!(answer.contains("not independently confirmed by the retrieved excerpts"));
        assert_eq!(grounded_answer("The speaker does not mention or address the moon landing.",0),
            "In the retrieved excerpts, the speaker does not mention or address the moon landing.");
    }
    #[test]
    fn streamed_sentences_match_the_final_template_answer() {
        let mut stream=SentenceStream::new(1);
        let mut spoken=Vec::new();
        for token in ["Answer: The account", " was called Baal [E1, E9]. The", " speaker", " repeats it.\nSco","pe: The speaker does not mention a bank statement.","\n\nQuestion: invented follow-up."] {
            spoken.extend(stream.push(token));
        }
        spoken.extend(stream.finish());
        assert_eq!(spoken,[
            "The account was called Baal [E1] (unavailable evidence reference).",
            "The speaker repeats it.",
            "In the retrieved excerpts, the speaker does not mention a bank statement.",
        ]);
    }
    #[test]
    fn quotes_absent_from_the_passages_are_flagged() {
        let passages = vec![
            "to:\nBaal. name: Wachovia Bank, N.A.\nPlease wire Eleven Thousand Four Hundred Thirty Eight Dollars froin the above account".to_string(),
            "So, the only time the account is named Baal is when it's actually named Baal.".to_string(),
            "the fact that this account name is actually baal on the patreon".to_string(),
        ];
        let answer = "The memo says \"Account Name: Baal\" and asks to \"wire Eleven Thousand Four Hundred Thirty Eight Dollars from the above account\". Voice A says \u{201c}the only time the account is named Baal\u{201d}, \"Baal\".";
        assert_eq!(mark_unverified_quotes(answer, &passages),
            "The memo says \"Account Name: Baal\" (not verbatim in the cited passages) and asks to \"wire Eleven Thousand Four Hundred Thirty Eight Dollars from the above account\". Voice A says \u{201c}the only time the account is named Baal\u{201d}, \"Baal\".");
        let mut stream = SentenceStream::new(0).with_passages(passages);
        assert_eq!(stream.push("It reads \"Account Name: Baal\" on screen. ").as_slice(), ["It reads \"Account Name: Baal\" (not verbatim in the cited passages) on screen."]);
    }
    #[test]
    fn a_long_quote_with_a_changed_meaning_word_is_flagged() {
        let passages = vec!["I dont know of any. they did a great job of closing most., at least for the goyim".to_string()];
        let altered = "\"they did a great job of closing most, at least for the globe\"";
        assert!(mark_unverified_quotes(altered, &passages).ends_with("(not verbatim in the cited passages)"));
        let exact = "\"they did a great job of closing most, at least for the goyim\"";
        assert_eq!(mark_unverified_quotes(exact, &passages), exact);
    }
    #[test]
    fn quotes_found_only_in_frame_descriptions_are_labeled() {
        let description = vec!["The email reads: they did a great job closing most at least for the globe.".to_string()];
        let answer = "It says \"did a great job closing most at least for the globe\".";
        assert_eq!(mark_quotes(answer, &[], &description), "It says \"did a great job closing most at least for the globe\" (from an automatic frame description; wording may be invented).");
    }
    #[test]
    fn short_fragments_join_and_the_remainder_is_flushed() {
        let mut stream=SentenceStream::new(0);
        assert!(stream.push("Yes. It is so. And").is_empty());
        assert_eq!(stream.push(" more follows. ").as_slice(),["Yes. It is so. And more follows."]);
        assert_eq!(stream.finish().as_deref(),None);
        let mut stream=SentenceStream::new(0);
        stream.push("Unfinished thought");
        assert_eq!(stream.finish().as_deref(),Some("Unfinished thought"));
    }
}
