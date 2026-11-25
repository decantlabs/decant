//! The file a side's subprocess hands back: one JSON line that describes the outputs (or the
//! failure), then the bytes of every subresource in order.

use serde::{Deserialize, Serialize};

use crate::compare::Output;

#[derive(Serialize, Deserialize)]
struct Head {
    error: Option<String>,
    outputs: Vec<Output>,
    lens: Vec<usize>,
}

/// Encodes what one side produced.
pub fn encode(result: &Result<Vec<Output>, String>) -> Vec<u8> {
    let (error, mut outputs) = match result {
        Ok(o) => (None, o.clone()),
        Err(e) => (Some(e.clone()), Vec::new()),
    };
    let mut lens = Vec::new();
    let mut body = Vec::new();
    for o in &mut outputs {
        for s in &mut o.subs {
            lens.push(s.bytes.len());
            body.append(&mut s.bytes);
        }
    }
    let mut out = serde_json::to_vec(&Head { error, outputs, lens }).expect("serializable");
    out.push(b'\n');
    out.extend_from_slice(&body);
    out
}

/// Decodes [`encode`]'s bytes. A malformed file is an error of the side.
pub fn decode(bytes: &[u8]) -> Result<Vec<Output>, String> {
    let nl = bytes.iter().position(|&b| b == b'\n').ok_or("side output has no header")?;
    let head: Head = serde_json::from_slice(&bytes[..nl]).map_err(|e| format!("side output header: {e}"))?;
    if let Some(e) = head.error {
        return Err(e);
    }
    let mut at = nl + 1;
    let mut lens = head.lens.iter();
    let mut outputs = head.outputs;
    for o in &mut outputs {
        for s in &mut o.subs {
            let n = *lens.next().ok_or("side output: too few lengths")?;
            s.bytes = bytes.get(at..at + n).ok_or("side output is truncated")?.to_vec();
            at += n;
        }
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{Elem, Sub};

    #[test]
    fn outputs_and_failures_round_trip() {
        let o = vec![Output {
            name: "u0".into(),
            texture: false,
            subs: vec![
                Sub { label: String::new(), width: 2, height: 1, depth: 1, record: vec![Elem::Auto], bytes: vec![1, 2, 3, 4, 5, 6, 7, 8] },
                Sub { label: "mip 1".into(), width: 1, height: 1, depth: 1, record: vec![Elem::F16, Elem::F16], bytes: vec![9, 10, 11, 12] },
            ],
        }];
        assert_eq!(decode(&encode(&Ok(o.clone()))).unwrap(), o);
        assert_eq!(decode(&encode(&Err("device removed".into()))), Err("device removed".to_string()));
        let mut cut = encode(&Ok(o));
        cut.pop();
        assert!(decode(&cut).is_err());
    }
}
