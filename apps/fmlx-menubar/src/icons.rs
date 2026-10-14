// Icons rendered from the bundled SVGs at runtime, so the artwork has one
// source: assets/menubar.svg (template glyph, the F-and-arrow mark of the
// admin UI without its background) and assets/app-icon.svg (the full logo).

use resvg::{tiny_skia, usvg};
use tray_icon::Icon;

const MENUBAR_SVG: &[u8] = include_bytes!("../assets/menubar.svg");
const APP_ICON_SVG: &[u8] = include_bytes!("../assets/app-icon.svg");

/// Rasterise an SVG into a square RGBA buffer of `size` pixels with `pad`
/// pixels of empty margin on each side.
pub fn render(svg: &[u8], size: u32, pad: u32, opacity: f32) -> Vec<u8> {
    let tree = usvg::Tree::from_data(svg, &usvg::Options::default()).expect("bundled svg parses");
    let mut pixmap = tiny_skia::Pixmap::new(size, size).expect("non-zero size");
    let inner = (size - 2 * pad) as f32;
    let view = tree.size();
    let scale = inner / view.width().max(view.height());
    let transform = tiny_skia::Transform::from_scale(scale, scale).post_translate(pad as f32, pad as f32);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    let mut data = pixmap.take();
    // tiny-skia stores premultiplied RGBA; NSImage via tray-icon wants
    // straight alpha. Demultiply, then apply the requested opacity.
    for px in data.chunks_exact_mut(4) {
        let a = px[3] as f32 / 255.0;
        if a > 0.0 {
            for c in &mut px[..3] {
                *c = ((*c as f32 / a).round()).min(255.0) as u8;
            }
        }
        px[3] = (px[3] as f32 * opacity).round() as u8;
    }
    data
}

/// Menu bar glyph: solid while the server is up, dimmed otherwise. Used as
/// a template image, so macOS tints it for light / dark menu bars.
pub fn menubar(serving: bool) -> Icon {
    let size = 44;
    let rgba = render(MENUBAR_SVG, size, 5, if serving { 1.0 } else { 0.4 });
    Icon::from_rgba(rgba, size, size).expect("valid icon buffer")
}

/// Full-colour app logo as PNG, for the .icns built by scripts/build.sh.
pub fn app_icon_png(size: u32) -> Vec<u8> {
    let tree = usvg::Tree::from_data(APP_ICON_SVG, &usvg::Options::default()).expect("bundled svg parses");
    let mut pixmap = tiny_skia::Pixmap::new(size, size).expect("non-zero size");
    // macOS icon grid: the artwork sits inside ~80% of the canvas.
    let pad = size as f32 * 0.1;
    let scale = (size as f32 - 2.0 * pad) / tree.size().width();
    let transform = tiny_skia::Transform::from_scale(scale, scale).post_translate(pad, pad);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    pixmap.encode_png().expect("png encode")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_has_ink_and_transparent_corners() {
        let px = render(MENUBAR_SVG, 44, 5, 1.0);
        assert_eq!(px.len(), 44 * 44 * 4);
        assert_eq!(px[3], 0, "top-left corner is transparent");
        assert!(px.chunks_exact(4).any(|p| p[3] == 255), "glyph is drawn");
    }

    #[test]
    fn dimmed_glyph_is_translucent() {
        let px = render(MENUBAR_SVG, 44, 5, 0.4);
        let max_alpha = px.chunks_exact(4).map(|p| p[3]).max().unwrap();
        assert!(max_alpha <= 103 && max_alpha > 90);
    }
}
