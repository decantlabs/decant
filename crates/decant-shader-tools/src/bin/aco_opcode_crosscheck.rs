//! Compares the opcode tables of the translator's GFX10 decoder with the GFX10 numbering of the
//! ACO compiler of Mesa. It replaces the Python script `aco_opcode_crosscheck.py`.
//!
//! Usage: `aco_opcode_crosscheck <decode folder> <aco_opcodes.cpp>`. The exit code is 1 when an
//! `ECART` exists. See [`decant_shader_tools::aco`] for the rules.

use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "Recoupe les tables d'opcodes du decodeur GFX10 du traducteur avec celles du compilateur ACO de Mesa.

usage : aco_opcode_crosscheck <dossier decode> <aco_opcodes.cpp>
  <dossier decode>  : src/graphics/shader/recompiler/frontend/decode du checkout des sources du traducteur
  <aco_opcodes.cpp> : le fichier que Mesa genere pendant sa compilation a partir de aco_opcodes.py
                      (src/amd/compiler/aco_opcodes_cpp.py, sans argument, sortie standard), pris dans un build
                      local de Mesa (Mesa est sous licence MIT, rien n'en est copie dans ce depot)

Chaque entree {numero, Opcode::NOM} des tables du decodeur est comparee au nom qu'ACO donne a ce numero pour GFX10 :
  identique : meme nom ;
  alias     : meme instruction sous un nom GFX10 different (v_add_nc_u32 = v_add_u32, cmp ne = lg, suffixe _e64...) ;
  absent    : ACO n'a aucune instruction GFX10 a ce numero (entree d'une autre generation, jamais emise pour GFX10) ;
  ECART     : ACO donne une autre instruction a ce numero. Code de sortie 1 s'il y en a au moins un.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    }
    match decant_shader_tools::aco::run(Path::new(&args[0]), Path::new(&args[1])) {
        Ok((report, ecart)) => {
            print!("{report}");
            if ecart { ExitCode::FAILURE } else { ExitCode::SUCCESS }
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
