//! The cases of the former `tests/test_gdk_config.cpp`, on the same synthetic config (invented identity and
//! ids) and hand crafted header buffers (no real images), plus the exact error texts.

use super::*;

const SYNTHETIC: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<!-- Invented values only. -->
<Game configVersion="1">
  <Identity Name="Contoso.SyntheticTitle" Publisher="CN=Contoso &amp; Co" Version="1.2.3.4"/>
  <ExecutableList>
    <Executable Name="Tool.exe" IsDevOnly="true" TargetDeviceFamily="Scarlett"/>
    <Executable Name="Synthetic.exe" TargetDeviceFamily="Scarlett" Id="Game"/>
    <Executable Name="SyntheticPC.exe" TargetDeviceFamily="PC"/>
  </ExecutableList>
  <TitleId>0ABC1234</TitleId>
  <StoreId>9PZZZZZZZZZZ</StoreId>
  <MSAAppId>0000000000000001</MSAAppId>
  <ShellVisuals DefaultDisplayName="Synthetic" PublisherDisplayName="Contoso"/>
  <PersistentLocalStorage>
    <SizeMB>1024</SizeMB>
    <GrowableToMB>2048</GrowableToMB>
  </PersistentLocalStorage>
</Game>
"#;

/// A config whose only executable is a dev only tool. The library records the fallback, the
/// `--exec` command line refuses it.
const DEV_ONLY_ONLY: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<!-- Invented values only. -->
<Game configVersion="1">
  <Identity Name="Contoso.SyntheticTitle" Publisher="CN=Contoso" Version="1.2.3.4"/>
  <ExecutableList>
    <Executable Name="Tool.exe" IsDevOnly="true" TargetDeviceFamily="Scarlett"/>
  </ExecutableList>
</Game>
"#;

fn read(xml: &str) -> Result<GdkConfig, String> {
    read_gdk_config_from_bytes(xml.as_bytes())
}

#[test]
fn synthetic_config() {
    let c = read(SYNTHETIC).unwrap();
    assert_eq!((c.config_version.as_str(), c.version.as_str()), ("1", "1.2.3.4"));
    assert_eq!((c.name.as_str(), c.publisher.as_str()), ("Contoso.SyntheticTitle", "CN=Contoso & Co"));
    assert_eq!((c.title_id.as_str(), c.store_id.as_str(), c.msa_app_id.as_str()), ("0ABC1234", "9PZZZZZZZZZZ", "0000000000000001"));
    assert_eq!(c.game_os_version, "");
    assert_eq!(c.executables.len(), 3);
    assert!(c.executables[0].is_dev_only);
    // The dev only tool is listed first: the primary must still be the other one.
    assert_eq!(c.primary_executable, "Synthetic.exe");
    assert_eq!(c.executables[1].target_device_family, "Scarlett");
    assert_eq!(c.persistent_local_storage_mb, Some(1024));
    assert_eq!(read(DEV_ONLY_ONLY).unwrap().persistent_local_storage_mb, None);
}

#[test]
fn title_ids_and_the_data_folder() {
    assert_eq!(title_id_number(" 0x0ABC1234 "), Some(0x0ABC_1234));
    assert_eq!(title_id_number("0abc1234"), Some(0x0ABC_1234));
    assert_eq!(title_id_number("title"), None);
    if let Some(folder) = persistent_local_storage_folder(0xABC) {
        assert!(folder.ends_with(Path::new("Decant").join("00000ABC").join("pls")), "{}", folder.display());
    }
}

#[test]
fn dev_only_fallback() {
    let c = read(DEV_ONLY_ONLY).unwrap();
    assert_eq!(c.executables.len(), 1);
    assert!(c.executables[0].is_dev_only);
    assert_eq!(c.primary_executable, "Tool.exe");
}

#[test]
fn refusals_with_their_texts() {
    let cases = [
        ("<Package/>", "root element is <Package>, expected <Game>"),
        (r#"<Game><Identity Name="a" Publisher="b" Version="1.0"/></Game>"#, "Identity Version is not a quad: 1.0"),
        (r#"<Game><Identity Name="a" Publisher="b"/><ExecutableList/></Game>"#, "ExecutableList has no Executable entries"),
        (r#"<Game><Identity Name="a""#, "unterminated tag <Identity>"),
        (r#"<Game><Identity Name="a"/></Game>"#, "Identity Name and Publisher are required"),
        ("", "expected exactly one root element"),
        ("<a/><b/>", "expected exactly one root element"),
        ("text", "unexpected text at top level"),
        ("</Game>", "unexpected closing tag"),
        ("< Game/>", "empty element name"),
        ("<Game =\"1\"/>", "bad attribute in <Game>"),
        ("<Game a/>", "expected '=' after a"),
        ("<Game a=1/>", "expected quoted value for a"),
        ("<Game a=\"1/>", "unterminated attribute a"),
        ("<Game/x>", "bad self-closing tag"),
        ("<Game>", "unterminated <Game>"),
        ("<Game><!-- x </Game>", "unterminated comment"),
        ("<Game></Game x>", "bad closing tag"),
        ("<Game></Gamex>", "mismatched </Gamex> for <Game>"),
    ];
    for (xml, error) in cases {
        assert_eq!(read(xml).unwrap_err(), error, "{xml}");
    }
}

#[test]
fn tokenizer_details() {
    let xml = "\u{feff}<?xml version='1.0'?>\n<!-- a -->\n<Game configVersion='0'>\
        <Identity Name='N&lt;&gt;&quot;&apos;&unknown;&amp' Publisher = \"P\"></Identity>\
        <TitleId> \t A<!-- c -->B &amp; C\r\n</TitleId>\
        <GameOsVersion>1.2</GameOsVersion>\
        <ExecutableList><Other Name='x'/><Executable Name=''/><Executable Name='A.exe' IsDevOnly='1'/>\
        <Executable Name='B.exe' IsDevOnly='TRUE'/></ExecutableList></Game><!-- end -->\n";
    let c = read(xml).unwrap();
    assert_eq!(c.config_version, "0");
    assert_eq!(c.name, "N<>\"'&unknown;&amp");
    assert_eq!(c.version, "1.0.0.0");
    assert_eq!(c.title_id, "AB & C");
    assert_eq!(c.game_os_version, "1.2");
    let names: Vec<_> = c.executables.iter().map(|e| (e.name.as_str(), e.is_dev_only)).collect();
    assert_eq!(names, [("A.exe", true), ("B.exe", false)]);
    assert_eq!(c.primary_executable, "B.exe");
}

#[test]
fn version_quads() {
    for (v, ok) in [("1.2.3.4", true), ("10.0.0.65535", true), ("1.2.3", false), ("1.2.3.4.5", false), ("1.2.3.", false), ("1..3.4", false), ("a.2.3.4", false), ("", false)] {
        assert_eq!(is_version_quad(v.as_bytes()), ok, "{v}");
    }
}

/// DOS header with `e_lfanew` at 0x80, `PE\0\0`, COFF header with `machine`, PE32+ optional
/// header (0xF0 bytes) with `subsystem`. Zeros elsewhere are fine for the header check.
fn crafted_pe(machine: u16, subsystem: u16) -> Vec<u8> {
    let mut buf = vec![0u8; 0x200];
    buf[..2].copy_from_slice(b"MZ");
    buf[0x3c] = 0x80;
    let pe = 0x80;
    buf[pe..pe + 2].copy_from_slice(b"PE");
    buf[pe + 4..pe + 6].copy_from_slice(&machine.to_le_bytes());
    buf[pe + 20] = 0xf0;
    buf[pe + 24..pe + 26].copy_from_slice(&0x20bu16.to_le_bytes());
    buf[pe + 24 + 68..pe + 24 + 70].copy_from_slice(&subsystem.to_le_bytes());
    buf
}

#[test]
fn pe_preflight() {
    assert_eq!(check_pe_image_from_memory(&crafted_pe(0x8664, 2)), Ok(()));
    assert_eq!(check_pe_image_from_memory(&crafted_pe(0x8664, 3)), Ok(()));
    let pe = crafted_pe(0x8664, 2);
    let mut no_mz = pe.clone();
    no_mz[..2].copy_from_slice(b"NE");
    let mut no_sig = pe.clone();
    no_sig[0x81] = b'X';
    let mut low_offset = pe.clone();
    low_offset[0x3c] = 0x3f;
    let mut pe32 = pe.clone();
    pe32[0x80 + 24] = 0x0b;
    pe32[0x80 + 25] = 0x01;
    let cases: [(&[u8], &str); 10] = [
        (&crafted_pe(0x014c, 2), "machine 0x14c is not AMD64 (0x8664)"),
        (&crafted_pe(0, 2), "machine 0 is not AMD64 (0x8664)"),
        (&pe[..0x30], "truncated: 48 bytes is too small for a DOS header"),
        (&pe[..0x90], "PE header offset is out of range"),
        (&pe[..0xa0], "optional header is truncated"),
        (&no_mz, "no MZ signature"),
        (&no_sig, "no PE\\0\\0 signature at e_lfanew"),
        (&low_offset, "PE header offset is out of range"),
        (&pe32, "optional header magic 0x10b is not PE32+ (0x20b)"),
        (&crafted_pe(0x8664, 1), "subsystem 0x1 is not Windows GUI/CUI"),
    ];
    for (data, error) in cases {
        assert_eq!(check_pe_image_from_memory(data).unwrap_err(), error);
    }
}

#[test]
fn pe_files() {
    let dir = std::env::temp_dir().join(format!("gdk-config-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let good = dir.join("good.exe");
    let empty = dir.join("empty.exe");
    std::fs::write(&good, crafted_pe(0x8664, 2)).unwrap();
    std::fs::write(&empty, b"").unwrap();
    assert_eq!(check_pe_image(&good), Ok(()));
    assert_eq!(check_pe_image(&empty).unwrap_err(), format!("cannot read {}", empty.display()));
    let missing = dir.join("no-such-file-here.exe");
    assert_eq!(check_pe_image(&missing).unwrap_err(), format!("cannot open {}", missing.display()));
    std::fs::write(&good, crafted_pe(0x8664, 9)).unwrap();
    assert_eq!(check_pe_image(&good).unwrap_err(), format!("{}: subsystem 0x9 is not Windows GUI/CUI", good.display()));
    std::fs::remove_dir_all(&dir).unwrap();
}
