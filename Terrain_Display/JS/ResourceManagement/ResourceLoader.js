//  Initialize the resource loader.
function ResourceLoader(exitCallback, holderRef) {

    this.resourceCount = 0;
    this.RLStorage = new ResourceLoaderStorage();
    this.RLRequests = new ResourceLoaderStorage();
    this.loadCount = 0;
    this.exitCallback = exitCallback;
    this.holderRef = holderRef;

}

//  Add a request to the Resouce Loader
ResourceLoader.prototype.addResourceRequest = function (type, url) {

    var loadRequest = new LoadRequest();
    loadRequest.TYPE = type;
    loadRequest.URL = url;

    if (loadRequest.TYPE == "TEXT") {

        this.RLRequests.TEXT.push(loadRequest);
        loadRequest.INDEX = this.RLRequests.TEXT.length - 1;

    } else if (loadRequest.TYPE == "IMAGE") {

        this.RLRequests.IMAGE.push(loadRequest);
        loadRequest.INDEX = this.RLRequests.IMAGE.length - 1;

    } else if (loadRequest.TYPE == "JSON") {

        this.RLRequests.JSON.push(loadRequest);
        loadRequest.INDEX = this.RLRequests.JSON.length - 1;

    }

    //  console.log(this.RLRequests);

};

//  Start loading the requested resources.
ResourceLoader.prototype.loadRequestedResources = function () {

    for (var i = 0; i < this.RLRequests.TEXT.length; i++) {
        this.loadSingleResource(this.RLRequests.TEXT[i]);
    }

    for (var i = 0; i < this.RLRequests.IMAGE.length; i++) {
        this.loadSingleResource(this.RLRequests.IMAGE[i]);
    }

    for (var i = 0; i < this.RLRequests.JSON.length; i++) {
        this.loadSingleResource(this.RLRequests.JSON[i]);
    }

};


//  Load one resource.
ResourceLoader.prototype.loadSingleResource = function (loadRequest) {

    var RLReference = this;

    this.resourceCount++;

    if (loadRequest.TYPE == "TEXT") {

        this.RLStorage.TEXT.push(null);
        this.requestTEXTResource(RLReference, loadRequest);

    } else if (loadRequest.TYPE == "IMAGE") {

        this.RLStorage.IMAGE.push(null);
        this.requestIMAGEResource(RLReference, loadRequest);

    } else if (loadRequest.TYPE == "JSON") {

        this.RLStorage.JSON.push(null);
        this.requestJSONResource(RLReference, loadRequest);

    }

};

//  Handles loading a text resource.
ResourceLoader.prototype.requestTEXTResource = function (RLReference, loadRequest) {
    console.log("TEXT Requested.");

    $.ajax({
        mimeType: 'text/plain; charset=x-user-defined',
        url: loadRequest.URL,
        type: 'GET',
        dataType: 'text',
        success: function (loadedData) {
            RLReference.storeLoadedData(loadRequest, loadedData)
        },
    });

};

//  Handles loading a image resource.
ResourceLoader.prototype.requestIMAGEResource = function (RLReference, loadRequest) {
    console.log("IMAGE Requested.");

    var image = new Image();
    image.onload = function () {
        RLReference.storeLoadedData(loadRequest, image);
    };

    image.onerror = function () {
        console.log("Cannot load image");
    };
    image.src = loadRequest.URL;

};


//  Handles loading a JSOM resource.
ResourceLoader.prototype.requestJSONResource = function (RLReference, loadRequest) {
    console.log("JSON Requested.");


    $.ajax({
        url: loadRequest.URL,
        type: 'GET',
        dataType: 'json',
        success: function (loadedData) {
            RLReference.storeLoadedData(loadRequest, loadedData)
        },
    });

};

//  Stores all the loaded data in the functions.
ResourceLoader.prototype.storeLoadedData = function (loadRequest, loadedData) {
    this.loadCount++;
    console.log(this.loadCount);
    console.log(loadRequest);

    if (loadRequest.TYPE == "TEXT") {
        this.RLRequests.TEXT[loadRequest.INDEX].isLOADED = true;
        this.RLStorage.TEXT[loadRequest.INDEX] = loadedData;
    } else if (loadRequest.TYPE == "IMAGE") {
        this.RLRequests.IMAGE[loadRequest.INDEX].isLOADED = true;
        this.RLStorage.IMAGE[loadRequest.INDEX] = loadedData;
    } else if (loadRequest.TYPE == "JSON") {
        this.RLRequests.JSON[loadRequest.INDEX].isLOADED = true;
        this.RLStorage.JSON[loadRequest.INDEX] = loadedData;
    }

    //    console.log(this.RLStorage.TEXT);
    //    console.log(this.RLStorage.IMAGE);
    //    console.log(this.RLStorage.JSON);

    this.checkAllLoadedData();

};

//  Runs a check to see if all the data is loaded.
ResourceLoader.prototype.checkAllLoadedData = function () {

    var allTEXTLoaded = true;
    var allJSONLoaded = true;
    var allIMAGELoaded = true;

    for (var i = 0; i < this.RLRequests.TEXT.length; i++) {
        if (this.RLRequests.TEXT[i] == null) {
            allTEXTLoaded = false;
        } else if (this.RLRequests.TEXT[i].isLOADED == false) {
            allTEXTLoaded = false;
        }

    }

    for (var i = 0; i < this.RLRequests.IMAGE.length; i++) {
        if (this.RLRequests.IMAGE[i] == null) {
            allIMAGELoaded = false;

        } else if (this.RLRequests.IMAGE[i].isLOADED == false) {
            allIMAGELoaded = false;
        }

    }

    for (var i = 0; i < this.RLRequests.JSON.length; i++) {
        if (this.RLRequests.JSON[i] == null) {
            allJSONLoaded = false;
        } else if (this.RLRequests.JSON[i].isLOADED == false) {
            allJSONLoaded = false;
        }

    }

    if (allTEXTLoaded && allJSONLoaded && allIMAGELoaded) {
        console.log("Completed Load Resources.");
        this.exitCallback(this.holderRef);
    } else {

    }
};

//  Just the definition of the storage.
function ResourceLoaderStorage() {
    this.TEXT = [];
    this.IMAGE = [];
    this.JSON = [];
}