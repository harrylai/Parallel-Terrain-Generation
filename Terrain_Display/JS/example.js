
var vTerrain = [];

//declare an array that holds normals per vertex
var normals = [];


function exampleLoad() {
    this.RL = null; // 
}

exampleLoad.prototype.loadResources = function () {

    this.RL = new ResourceLoader(this.resourcesLoaded, this);
    this.RL.addResourceRequest("TEXT", "JS/Assets/TEXT/default_vertex_shader.txt");
    this.RL.addResourceRequest("TEXT", "JS/Assets/TEXT/default_fragment_shader.txt");
    this.RL.addResourceRequest("TEXT", "JS/Assets/TEXT/vertex.txt")
    this.RL.loadRequestedResources();

};

exampleLoad.prototype.resourcesLoaded = function (exampleLoadReference) {
    //exampleLoadReference.completeCheck();
    exampleLoadReference.extractData();
    //exampleLoadReference.calculateNormals();
    console.log(vTerrain);
    start();

};

exampleLoad.prototype.completeCheck = function () {
    // console.log(this.RL.RLStorage.TEXT[0]);
    // console.log(this.RL.RLStorage.TEXT[1]);
    console.log(this.RL.RLStorage.TEXT[2]);
};

exampleLoad.prototype.extractData = function () {
    // Split the data by lines and process them one by one
    var temp;
    var lines = this.RL.RLStorage.TEXT[2].trim().split("\n");
    console.log(lines.length);
    for (var i = 0; i < lines.length; i++)
    {
        temp = lines[i].split(/\s+/);
        //console.log(temp);
            vTerrain.push(parseFloat(temp[0]));
            vTerrain.push(parseFloat(temp[1]));
            vTerrain.push(parseFloat(temp[2])/6);

    }
};
