import AI4Child from './js/AI4Child.react';
import NoWebcam from './js/NoWebcam.react';
import React from 'react';
import ReactDOM from 'react-dom';

// const CLASSIFIER_ENDPOINT = 'https://c3586iuyoe.execute-api.eu-west-1.amazonaws.com/Stage/uploadtos3';
const CLASSIFIER_ENDPOINT = 'http://localhost:5000/predict';
const SAMPLING_INTERVAL = 500;
// const SAMPLE_RESPONSE = "{'sightengine': {'status': 'success', 'request': {'id': 'req_3YmL8y5ZwsYOBUv2oySIm', 'timestamp': 1540050904.9663, 'operations': 1}, 'weapon': 0.6375, 'alcohol': 0.001, 'drugs': 0.002, 'media': {'id': 'med_3YmLCFG5QaOki7mR2xmT8', 'uri': 'https://s3-eu-west-1.amazonaws.com/crimedetection/live/d9aa26ed-d471-11e8-b7c3-c521eb4347be.jpg'}}, 'azure': {'categories': [{'name': 'others_', 'score': 0.01171875}, {'name': 'people_', 'score': 0.54296875}, {'name': 'people_show', 'score': 0.34375}], 'color': {'dominantColorForeground': 'Black', 'dominantColorBackground': 'Black', 'dominantColors': ['Black'], 'accentColor': '16090A', 'isBwImg': False}, 'description': {'tags': ['person', 'clothing', 'man', 'standing', 'suit', 'wearing', 'holding', 'black', 'hand', 'dark', 'cutting', 'cut', 'woman', 'cake', 'dressed', 'young', 'white', 'knife'], 'captions': [{'text': 'a man wearing a suit and tie', 'confidence': 0.8962813890841713}]}, 'requestId': 'f2ee641f-45f0-4978-bba0-f0a7635317a0', 'metadata': {'width': 213, 'height': 176, 'format': 'Jpeg'}}}";
const SAMPLE_RESPONSE = '{"class_probabilities": {"nudity": 0.2, "violence": 0.2, "gambling": 0.2, "drugs": 0.2, "negative": 0.2}}';
const MAX_FRAMES_IN_ROLL = 15;
const MAX_CENSORED_FRAMES = 15;
const LABELS = ['knife', 'nudity', 'gun', 'blood'];
const VIOLENT = true;
const NON_VIOLENT = false;
const HIDE_THRESHOLD_AFTER = 300;

const videoRoot = $('video-root');
const video = $('video');
const videoLabels = $('video-labels');
const framesRoll = $('frames-roll');
const censoredFrames = $('censored-frames');
const thresholdSlider = $('threshold-slider');
const thresholdLabel = $('threshold-label');

let canvas;
let canvasCtx;
let lastViolentFrameId = 0;
let lastNonViolentFrameId = 0;
let hideTresholdLabelTimer = 0;


thresholdSlider.addEventListener('input', () => {
    thresholdLabel.textContent = getThreshold();
    thresholdLabel.classList.add('visible');
    window.clearTimeout(hideTresholdLabelTimer);
    hideTresholdLabelTimer = window.setTimeout(() => thresholdLabel.classList.remove('visible'), HIDE_THRESHOLD_AFTER);
}, false);

navigator.mediaDevices.getUserMedia({video: true})
    .then(stream => {
        $('app-root').classList.remove('loading');
        handleStream(stream);
    })
    .catch(e => {
        console.log(e);

        const appRoot = $('app-root');
        appRoot.classList.remove('loading');
        ReactDOM.render(<NoWebcam />, appRoot);
    });

function handleStream(stream) {
    show(video);
    video.srcObject = stream;

    setupCanvas();

    window.setInterval(sampleFrame, SAMPLING_INTERVAL);
}

function setupCanvas() {
    canvas = document.createElement('canvas');
    window.setTimeout(/*requestAnimationFrame(*/() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvasCtx = canvas.getContext('2d');
    }, 500);
}

function sampleFrame() {
    const frameId = Date.now();

    canvasCtx.drawImage(video, 0, 0);
    const base64Jpeg = canvas.toDataURL('image/jpeg');
    classifyFrame(frameId, base64Jpeg);
 
    prependToRoll(base64Jpeg, frameId);
}

function classifyFrame(frameId, frame, callback) {
    const prefix = 'data:image/jpeg;base64,';
    const base64Jpeg = frame.slice(prefix.length);

    // fetch(CLASSIFIER_ENDPOINT, {
    //     method: 'POST',
    //     headers: {
    //         'Accept': 'application/json',
    //         'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({image: base64Jpeg})
    // })
    //     .then(res => res.json())
    //     .then(res => {
    //         console.log(res);
    //         handleClassification(frameId, frame, res);
    //     });

    window.setTimeout(() => handleClassification(frameId, frame, SAMPLE_RESPONSE), 500);
}


function handleClassification(frameId, frame, prediction) {
    const [classification, labels] = getClassAndLabelsFromPrediction(JSON.parse(prediction));
    
    const miniature = $(miniatureIdFor(frameId));
    if (miniature) {
        if (classification === VIOLENT) {
            miniature.classList.add('violence-miniature');
        } else {
            miniature.classList.add('non-violence-miniature');
        }
    }

    if (classification === VIOLENT) {
        if (!isInViolenceMode()) {
            if (frameId > lastNonViolentFrameId) {
                turnOnViolenceMode();
                applyLabels(labels);
            }
        } else {
            if (frameId > lastViolentFrameId) {
                applyLabels(labels);
            }
        }

        const censoredFrameTable = document.createElement('table');
        const censoredFrameRow = document.createElement('tr');
        const censoredFrameImgCell = document.createElement('td');
        const censoredFrameLabelsCell = document.createElement('td');

        const censoredFrame = document.createElement('img');
        censoredFrame.src = frame;

        censoredFrameImgCell.classList.add('censoredFrameImgRoot');
        
        censoredFrameLabelsCell.innerHTML = "&#8611; " + labels.join(', ');

        censoredFrameTable.classList.add('censoredFrameRoot');
        censoredFrameTable.appendChild(censoredFrameRow);
        censoredFrameRow.appendChild(censoredFrameImgCell);
        censoredFrameRow.appendChild(censoredFrameLabelsCell);
        censoredFrameImgCell.appendChild(censoredFrame);
        
        censoredFrameLabelsCell.classList.add('censoredFrameLabels');
        censoredFrames.insertBefore(censoredFrameTable, censoredFrames.firstChild);
        if (censoredFrames.childElementCount > MAX_CENSORED_FRAMES) {
            censoredFrames.removeChild(censoredFrames.lastChild);
        }

        lastViolentFrameId = frameId;
    } else {
        if (isInViolenceMode()) {
            if (frameId > lastViolentFrameId) {
                turnOffViolenceMode();
            }
        }
        
        lastNonViolentFrameId = frameId;
    }
}

function getClassAndLabelsFromPrediction(prediction) {
    const threshold = getThreshold();
    const classes = prediction['class_probabilities'];
    const labels = Object.keys(classes).filter(k => classes[k] > threshold);
    
    console.log(labels);

    return [labels.length > 0 ? VIOLENT : NON_VIOLENT, labels];
}

function getThreshold() {
    return thresholdSlider.value / 100;
}

function applyLabels(labels) {
    videoLabels.textContent = labels.join(', ');
}

function getRandomClassification(statusChangeProbability) {
    const currentlyShowingViolence = isInViolenceMode();

    return Math.random() <= statusChangeProbability
        ? !currentlyShowingViolence
        : currentlyShowingViolence;
}

function getRandomLabels() {
    const labelsCount = Math.floor(Math.random() * LABELS.length) + 1;
    const shuffledLabels = fisherYatesShuffle(LABELS);
    return shuffledLabels.slice(0, labelsCount);
}

function fisherYatesShuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function prependToRoll(base64Jpeg, frameId) {
    const miniature = document.createElement('img');
    miniature.setAttribute('id', miniatureIdFor(frameId));
    miniature.src = base64Jpeg;
    framesRoll.insertBefore(miniature, framesRoll.firstChild);
    if (framesRoll.childElementCount > MAX_FRAMES_IN_ROLL) {
        framesRoll.removeChild(framesRoll.lastChild);
    }
}

function miniatureIdFor(frameId) {
    return 'miniature-' + frameId;
}

function isInViolenceMode() {
    return video.classList.contains('blurred');
}

function turnOnViolenceMode() {
    videoRoot.classList.remove('non-violence');
    videoRoot.classList.add('violence');
    blurVideo();
}

function turnOffViolenceMode() {
    videoRoot.classList.remove('violence');
    videoRoot.classList.add('non-violence');
    unblurVideo();
}

function blurVideo() {
    video.classList.add('blurred');
}

function unblurVideo() {
    video.classList.remove('blurred');
}

function show(el) {
    el.classList.remove('hidden');
}

function $(id) {
    return document.getElementById(id);
}
