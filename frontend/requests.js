
const baseUrl = 'http://localhost:5000'

function setStatus(message) {
    document.getElementById('resource-status').innerHTML = message;
}

function loadResource(url, data) {
    return new Promise(function(resolve, reject) {
        const req = new XMLHttpRequest();
        req.open('POST', url);
        req.setRequestHeader('Content-Type', 'application/json');
        req.onload = function() {
            if(req.status == 200) {
                resolve(JSON.parse(req.responseText));
            } else {
                reject('Failed to load training session.!');
            }
        }
        req.send(JSON.stringify(data))
    });
}

async function getTrainingSessions() {
    const url = baseUrl + '/get-sessions';
    resp = await loadResource(url, {});
    const e = document.getElementById('training-session');
    for(let session of resp.sessions) {
        const option = document.createElement('option');
        option.text = session;
        e.add(option);
    }
}

async function loadTrainingSession(name) {
    const url = baseUrl + '/load-session';
    setStatus('Loading results of session ' + name);
    resp = await loadResource(url, {name: name});
    setStatus(resp.status);
}


document.body.onload = getTrainingSessions;
document.getElementById('training-session').onchange = function(e) {
    const name = e.target.value;
    console.log(name);
    if(name != '')
        loadTrainingSession(name);
}
