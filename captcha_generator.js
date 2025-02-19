const svgCaptcha = require('./svg-captcha');
const random = require('./svg-captcha/lib/random');
const sharp = require('sharp');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');


random.greyColor = function(min = 0, max = 2) {
  const int = Math.floor(Math.random() * (max - min + 1)) + min; 
  const hex = int.toString(16).padStart(2, '0');
  return `#${hex}${hex}${hex}`; 
};

random.color = function(bgColor) {
  return random.greyColor(); 
};


function generateCaptcha() {
  const captcha = svgCaptcha.create({
    size: 5,
    ignoreChars: 'oi',
    noise: 2,
    color: true,
    background: '#FFFFFF',
    width: 150,
    height: 50,
    charPreset: '0123456789',
    fontSize: 100
  });

  return { svg: captcha.svg, boxes: captcha.boxes, text: captcha.text };
}



let numberCount = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0 };
const numberUsageFile = './numberUsage.json';
if (fs.existsSync(numberUsageFile)) {
  try {
    numberCount = JSON.parse(fs.readFileSync(numberUsageFile, 'utf-8'));
  } catch (error) {
    console.log("Error parsing JSON, initializing defaults", error);
    
    numberCount = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0 };
  }
} else {
  numberCount = { '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0 };
}



const minimumCount = 2000;
let totalCaptchasGenerated = 0;


function allNumbersSufficientlyCovered() {
  return Object.values(numberCount).every(count => count >= minimumCount);
}


async function saveCaptchaAsJpeg() {
  
  const directory = './generated_captchas';
  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory);
  }

  
  while (!allNumbersSufficientlyCovered()) {
    const { svg, boxes, text } = generateCaptcha();

    
    text.split('').forEach(char => {
      if (numberCount.hasOwnProperty(char)) {
        numberCount[char]++;
      }
    });

    
    const jpgBuffer = await sharp(Buffer.from(svg, 'utf-8'))
      .jpeg({ quality: 100 })
      .toBuffer();

    const noisyBuffer = await addNoise(jpgBuffer);

    
    const imageFileName = `${directory}/${text}.jpg`;
    fs.writeFileSync(imageFileName, noisyBuffer);

    
    const labels = boxes.map(box => ({
      class: box.label,
      x_center: box.x_center,
      y_center: box.y_center,
      width: box.width,
      height: box.height
    }));

    const labelsFileName = `${directory}/${text}.json`;
    fs.writeFileSync(labelsFileName, JSON.stringify(labels, null, 2), 'utf-8');

    totalCaptchasGenerated++;



    
    fs.writeFileSync(numberUsageFile, JSON.stringify(numberCount, null, 2), 'utf-8');
  }

  console.log('All numbers used at least 1000 times!');
}


async function addNoise(imageBuffer) {
  const image = await loadImage(imageBuffer);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < imageData.data.length; i += 4) {
    const noise = Math.random() * 50 - 35; 

    if (Math.random() < 0.01) { 
      imageData.data[i] = 0;
      imageData.data[i + 1] = 0;
      imageData.data[i + 2] = 0;
    } else {
      imageData.data[i] = Math.min(255, Math.max(0, imageData.data[i] + noise));
      imageData.data[i + 1] = Math.min(255, Math.max(0, imageData.data[i + 1] + noise));
      imageData.data[i + 2] = Math.min(255, Math.max(0, imageData.data[i + 2] + noise));
    }

    imageData.data[i + 3] = 255; 
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.toBuffer('image/jpeg');
}


if (require.main === module) {
  saveCaptchaAsJpeg();
}
