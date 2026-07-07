![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# CUVIS.AI SAM3

This repository provides a port of SAM3 as a cuvis.ai plugin, enabling promptable segmentation and tracking pipelines. It is maintained by Cubert GmbH as part of the cuvis.ai ecosystem.

## Capabilities

This is a cuvis.ai plugin that adapts SAM3 for cuvis.ai stream-mode applications. It brings the following capabilities to cuvis.ai:

**Nodes**

- `SAM3TextPropagation`: text or concept-based object detection and tracking
- `SAM3BboxPropagation`: bounding-box prompt tracking
- `SAM3PointPropagation`: point prompt tracking
- `SAM3MaskPropagation`: mask or label-map based tracking
- `SAM3SegmentEverything`: prompt-free instance segmentation on a single frame
- `SAM3PointExpansion`: expands positive/negative click points into one object mask on a single frame

## Quick Start

For local development in this repository:

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git
cd cuvis-ai-sam3
uv sync --all-extras
```

For cuvis.ai usage examples, see the SAM3 object-tracking pipelines in [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai/tree/main/examples/object_tracking/sam3).

For the original upstream SAM3 repository README, installation details, research background, and example notebooks, see [README_original.md](README_original.md).

## Links

- **Documentation:** https://docs.cuvis.ai/latest/
- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai-sam3/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Original SAM3 README:** [README_original.md](README_original.md)

---

See [LICENSE](LICENSE) for repository licensing details.
