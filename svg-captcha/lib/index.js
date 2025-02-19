'use strict';
const chToPath = require('./ch-to-path');
const random = require('./random');
const optionMngr = require('./option-manager');

const opts = optionMngr.options;

const getLineNoise = function (width, height, options) {
	const hasColor = options.color;
	const noiseLines = [];
	const min = options.inverse ? 7 : 1;
	const max = options.inverse ? 15 : 9;
	let i = -1;

	while (++i < options.noise) {
		const start = `${random.int(1, 21)} ${random.int(1, height - 1)}`;
		const end = `${random.int(width - 21, width - 1)} ${random.int(1, height - 1)}`;
		const mid1 = `${random.int((width / 2) - 21, (width / 2) + 21)} ${random.int(1, height - 1)}`;
		const mid2 = `${random.int((width / 2) - 21, (width / 2) + 21)} ${random.int(1, height - 1)}`;
		const color = hasColor ? random.color() : random.greyColor(min, max);
		noiseLines.push(`<path d="M${start} C${mid1},${mid2},${end}" stroke="${color}" fill="none"/>`);
	}

	return noiseLines;
};

const getText = function (text, width, height, options) {
  const len = text.length;
  const spacing = (width - 2) / (len + 1);
  const min = options.inverse ? 10 : 0;
  const max = options.inverse ? 14 : 4;
  let i = -1;
  const out = [];
  const boxes = [];  // Rename 'labels' to 'boxes' for clarity

  while (++i < len) {
    const x = spacing * (i + 1) + Math.random() * (20 - 5) + 5;
    const y = (height / 2) + Math.random() * (5 - -5) - 5;
    const randomFontSize = options.fontSize + Math.random() * 10 - 5;
    const charPath = chToPath(text[i], Object.assign({ x, y, fontSize: randomFontSize }, options));

    const color = options.color ?
      random.color(options.background) : random.greyColor(min, max);

    const glyph = options.font.charToGlyph(text[i]);
    const charWidth = glyph.advanceWidth * (randomFontSize / options.font.unitsPerEm);
    const charHeight = (options.ascender + options.descender) * (randomFontSize / options.font.unitsPerEm);

    // Calculate bounding box center and dimensions
    const x_center = x;
    const y_center = y;
    const width_box = charWidth;
    const height_box = charHeight;

    // Add bounding box
    boxes.push({
      label: text[i],
      x_center: x_center / width,  // Normalize x_center to [0, 1]
      y_center: y_center / height,  // Normalize y_center to [0, 1]
      width: width_box / width,    // Normalize width to [0, 1]
      height: height_box / height  // Normalize height to [0, 1]
    });

    out.push(`<path fill="${color}" d="${charPath}"/>`);
  }

  return { svg: out.join(''), boxes };  // Return both svg and boxes
};




const fs = require('fs');
const createCaptcha = function (text, options) {
  text = text || random.captchaText();
  options = Object.assign({}, opts, options);
  const width = options.width;
  const height = options.height;
  const bg = options.background;
  if (bg) {
    options.color = true;
  }

  const bgRect = bg ?
    `<rect width="100%" height="100%" fill="${bg}"/>` : '';
  const { svg, boxes } = getText(text, width, height, options);

  const start = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0,0,${width},${height}">`;
  const xml = `${start}${bgRect}${svg}</svg>`;

  // Save the SVG as a file
  fs.writeFileSync('captcha_image.svg', xml);

  // Save the labels in YOLO format to a text file
  const labelsText = boxes.map(box =>
    `${box.label} ${box.x_center} ${box.y_center} ${box.width} ${box.height}`).join('\n');
  fs.writeFileSync('captcha_labels.txt', labelsText);

  return { svg: xml, boxes, text };
};

const create = function (options) {
  const text = random.captchaText(options);
  const data = createCaptcha(text, options);

  return data; // Return svg, boxes, and text
};


const createMathExpr = function (options) {
	const expr = random.mathExpr(options.mathMin, options.mathMax, options.mathOperator);
	const text = expr.text;
	const data = createCaptcha(expr.equation, options);

	return {text, data};
};

module.exports = createCaptcha;
module.exports.randomText = random.captchaText;
module.exports.create = create;
module.exports.createMathExpr = createMathExpr;
module.exports.options = opts;
module.exports.loadFont = optionMngr.loadFont;
