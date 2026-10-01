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
    ] {
        let mut from=0;
        while let Some(index)=output[from..].to_ascii_lowercase().find(phrase) {
            let start=from+index;let replacement=if output.as_bytes()[start].is_ascii_uppercase(){let mut r=replacement.to_string();r.get_mut(..1).unwrap().make_ascii_uppercase();r}else{replacement.to_string()};output.replace_range(start..start+phrase.len(),&replacement);from=start+replacement.len();
        }
    }
    output
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
    }
}
