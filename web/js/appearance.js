import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    reduxfinetune: { nodeColor: "#222e40", nodeBgColor: "#364254", width: 340},
};

const NODE_COLORS = {
    "ReduxFineTune": "reduxfinetune",
    "ReduxFineTuneAdvanced": "reduxfinetune",
    "ClipVision": "reduxfinetune",
    "ClipVisionStyleLoader": "reduxfinetune",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80]; // default size if not set
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "reduxfinetune.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);
