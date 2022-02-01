
let files = null;


function performValidation(id, numeric, image) {
    let val = document.getElementById(id).value;

    if (val === null || val == "" || val === "choose...") {
        return 1;
    }

    if (numeric) {
        if (isNaN(val) || val < 0 || val > 100) {
            return 1;
        }
    }

    if (image) {
        let parts = val.split(".");
        if (parts.length < 2) {
            return 1
        }
        let extension = parts[parts.length - 1].toLowerCase();
        if (["jpg", "jpeg", "png"].indexOf(extension) === -1) {
            return 1
        }

        files = document.getElementById(id).files;
        if (files.length != 1) {
            return 1;
        }

    }
    return 0;
}

function performValidationAndSetClass(id, numeric, image) {
    let ret = performValidation(id, numeric, image);

    document.getElementById(id).classList.add("is-invalid");
    document.getElementById(id).classList.add("is-valid");
    if (ret === 1) {
        document.getElementById(id).classList.remove("is-valid");
    } else {
        document.getElementById(id).classList.remove("is-invalid");
    }
    return ret
}

function performValidations() {
    let error = 0;
    error += performValidationAndSetClass("ageInput", true, false)
    error += performValidationAndSetClass("sexInput", false, false)
    error += performValidationAndSetClass("location", false, false)
    error += performValidationAndSetClass("formFile", false, true)
    return error
}

function buildParams(filekey) {
    let params = {
        "image_key": filekey,
        "age": document.getElementById("ageInput").value,
        "sex": document.getElementById("sexInput").value,
        "location": document.getElementById("location").value,
    }

    return serialize(params);

}

function getColorAndAlert(percentile){
    if (percentile < 75){
        return {"class": "alert alert-success", "text": "You are at a low risk setting. You should see a doctor if you \
         have any other reason that might present risks (family history, pain, physical ailments, etc)."}
    } else if (percentile < 95){
        return {"class": "alert alert-warning", "text": "You are at a medium risk setting. You should consider seeing a \
        doctor and with urgency if you present other risks (family history, pain, physical ailments, etc)."}
    } else {
        return {"class": "alert alert-danger", "text": "You are at a high risk setting. You should consider seeing a \
        doctor inmediately."}
    }
}

function handleReturnFromApiCall(data) {
    data = data['predicted_label'];
    let raw_score = (data['pred_raw'] * 100.0).toFixed(2)
    let percentile = (data['pred_percentile'] * 100.0).toFixed(0)

    document.getElementById("loading").hidden = true

    document.getElementById("score").innerText = raw_score
    document.getElementsByClassName("percentile").item(0).innerText = percentile
    document.getElementsByClassName("percentile").item(1).innerText = percentile

    document.getElementById("results-container").hidden = false


    let ret = getColorAndAlert(percentile);

    document.getElementById("advice").innerText = ret["text"]
    document.getElementById("result-alert").className = ret['class']

}

function handleFailureApi(data){
    console.log(data);
    document.getElementById("error-api").hidden = false;

    document.getElementById("loading").hidden = true;
    document.getElementById("results-container").hidden = true
}

function handleFileUploaded(filekey) {
    console.log("File uploaded succesfully: " + filekey );
    let params = buildParams(filekey);

    let endpoint = myEndpoint + params;

    let r = fetch(endpoint, {
        method: "GET"
    }).then((response) => {
            return response.json();
        }).then((data) => {
            handleReturnFromApiCall(data);
        }).catch((err) => {
            handleFailureApi(err)
        });
}

function uploadFile() {
    let file = document.getElementById("formFile").files[0]
    var fileName = file.name;

    var photoKey = Math.floor(Date.now() / 1000) + "_" + fileName;

    // Use S3 ManagedUpload class as it supports multipart uploads
    var upload = new AWS.S3.ManagedUpload({
        params: {
            Bucket: albumBucketName,
            Key: photoKey,
            Body: file
        }
    });

    var promise = upload.promise();

    promise.then(
        function (data) {
            handleFileUploaded(photoKey)
        },
        function (err) {
            handleFailureApi(err);
        }
    );

}

function handleSubmit() {

    document.getElementById("error-api").hidden = true

    let error = performValidations();
    if (error > 0) {
        return;
    }
    // @todo: freeze inputs
    document.getElementById("loading").hidden = false
    document.getElementById("results-container").hidden = true
    uploadFile()
}

serialize = function (obj) {
    var str = [];
    for (var p in obj)
        if (obj.hasOwnProperty(p)) {
            str.push(encodeURIComponent(p) + "=" + encodeURIComponent(obj[p]));
        }
    return str.join("&");
}

var fullEndpoint = "https://g2z896k26i.execute-api.us-east-1.amazonaws.com/default/melanoma?image_key=input_images/test.jpg&age=10&sex=female&location=torso"
var myEndpoint = "https://g2z896k26i.execute-api.us-east-1.amazonaws.com/default/melanoma?"

var albumBucketName = "melanoma-classification";
var bucketRegion = "us-east-1";
var IdentityPoolId = "us-east-1:44a2bb08-98d0-413e-b164-86cab1f7fe85";

AWS.config.update({
    region: bucketRegion,
    credentials: new AWS.CognitoIdentityCredentials({
        IdentityPoolId: IdentityPoolId
    })
});

var s3 = new AWS.S3({
    apiVersion: "2006-03-01",
    params: { Bucket: albumBucketName }
});

//@todo: send things every 30 seconds

function scheduleWarmUp(data){
    if (data["ok"] != true){
        console.log(data);
    }

    setTimeout(warmUp, 30 * 1000);
}

function warmUp(){

    console.log("Running warmup...")
    p = fetch(fullEndpoint, {
        method: "GET"
    })


    x = p.then((response) => {
        return response.json();
    }).then((data) =>{
        console.log("Warmup succesful");
        return data
    })


    x = p.catch((response) => {
        console.log("Problem with warmup: " + response);
    })

    x.then((data) => scheduleWarmUp(data));
}

warmUp()