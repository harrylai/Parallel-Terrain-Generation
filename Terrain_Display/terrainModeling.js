//-------------------------------------------------------------------------
function terrainFromIteration(n, minX,maxX,minY,maxY, vertexArray, faceArray,normalArray)
{
    var deltaX=(maxX-minX)/n;
    var deltaY=(maxY-minY)/n;
    for(var i=0;i<=n;i++)
       for(var j=0;j<=n;j++)
       {
           // vertexArray.push(minX+deltaX*j);
           // vertexArray.push(minY+deltaY*i);
           // vertexArray.push(0);
             
             normalArray.push(0);
             normalArray.push(0);
             normalArray.push(0);
       }
 //   for(var i=0; i<30; i++)
      console.log("yes we can")
      // generateTerrainVertices(vertexArray, minX,maxX,minY,maxY,n);
    /*
    var stepX = maxX-minX;
    var stepY = maxY-minY;
    
    vertexArray[2] = 0.39;
    vertexArray[3*n+2] = 0.5;
    vertexArray[3*(n*(n+1))+2] = 0.3;
    vertexArray[3*(n + n*(n+1))+2] = 0.35;
    while (stepX>= 2*deltaX && stepY >= 2*deltaY){
      for(var j=0; j< (maxY - minY)/stepY; j++){
        for(var i=0; i< (maxX - minX)/stepX; i++){
            var offsetX= stepX/deltaX;
            var offsetY= (stepY/deltaY)*(n+1); 
          //do diamond step
            var startindex = j*(stepY/deltaY)*(n+1) + i*(stepX/deltaX);
            var sum = vertexArray[3*(startindex)+2] + vertexArray[3*(startindex + offsetX)+2] + vertexArray[3*(startindex + offsetY)+2] + vertexArray[3*(startindex + offsetX + offsetY)+2];
            var avg = Math.random()*0.05+(sum/4);
            vertexArray[3*(startindex + offsetX/2 + offsetY/2) +2] = avg;
          //do square step
            var sum1 = avg + vertexArray[3*(startindex)+2] + vertexArray[3*(startindex + offsetX)+2];
            var avg1 = Math.random()*0.05 + (sum1 /3);
            vertexArray[3*(startindex + offsetX/2)+2] = avg1;
            
            var sum2 = avg + vertexArray[3*(startindex)+2] + vertexArray[3*(startindex + offsetY)+2];
            var avg2 = Math.random()*0.05 + (sum2 /3);
            vertexArray[3*(startindex + offsetY/2)+2] = avg2;
          
            var sum3 = avg + vertexArray[3*(startindex + offsetY)+2] + vertexArray[3*(startindex + offsetX + offsetY)+2];
            var avg3 = Math.random()*0.05 + (sum3 /3);
            vertexArray[3*(startindex + offsetX/2 + offsetY)+2] = avg3;
          
            var sum4 = avg + vertexArray[3*(startindex + offsetX + offsetY)+2] + vertexArray[3*(startindex + offsetX)+2];
            var avg4 = Math.random()*0.05 + (sum4 /3);
            vertexArray[3*(startindex + offsetX + offsetY/2)+2] = avg4;
        }
      }
      stepX=stepX/2;
      stepY=stepY/2;
    }
    */
    var numT=0;
    for(var i=0;i<n;i++)
       for(var j=0;j<n;j++)
       {
           var vid = i*(n+1) + j;
           faceArray.push(vid);
           faceArray.push(vid+1);
           faceArray.push(vid+n+1);
           
           faceArray.push(vid+1);
           faceArray.push(vid+1+n+1);
           faceArray.push(vid+n+1);
           numT+=2;
       }
	for(var j=0; j<n+1; j++){
		for( var i=0; i<n+1; i++){
			var startindex = i+j*(n+1);
			var v1 = vec3.fromValues(vertexArray[3*(startindex)], vertexArray[3*(startindex)+1], vertexArray[3*(startindex)+2]);
			var v2 = vec3.fromValues(vertexArray[3*(startindex+n+1)], vertexArray[3*(startindex+n+1)+1], vertexArray[3*(startindex+n+1)+2]);
			var v3 = vec3.fromValues(vertexArray[3*(startindex+1)], vertexArray[3*(startindex+1)+1], vertexArray[3*(startindex+1)+2]);
			var edge1 = vec3.create();
			var edge2 = vec3.create();
			var normal = vec3.create();
			//get two edges for normal vector
			vec3.subtract(edge1, v1, v2);
			vec3.subtract(edge2, v3, v2);
			vec3.cross(normal, edge1, edge2);
			//vec3.normalize(normal, normal);
			//push results to normal array
			normalArray[3*(startindex)] += normal[0];
      normalArray[3*(startindex)] += normal[1];
      normalArray[3*(startindex)] += normal[2];

      normalArray[3*(startindex+n+1)] += normal[0];
      normalArray[3*(startindex+n+1)] += normal[1];
      normalArray[3*(startindex+n+1)] += normal[2];

      normalArray[3*(startindex+1)] += normal[0];
      normalArray[3*(startindex+1)] += normal[1];
      normalArray[3*(startindex+1)] += normal[2];
		}
	}

  for(var j=0; j<n+1; j++){
    for( var i=0; i<n+1; i++){
      var normal = vec3.create();
      var startindex = i+j*(n+1);
      normal = vec3.fromValues(vertexArray[3*(startindex)], vertexArray[3*(startindex)+1], vertexArray[3*(startindex)+2]);
      normalArray[3*(startindex)] += normal[0];
      normalArray[3*(startindex)+1] += normal[1];
      normalArray[3*(startindex)+2] += normal[2];
    } 
  }
    return numT;
}

//-------------------------------------------------------------------------
function generateLinesFromIndexedTriangles(faceArray,lineArray)
{
    numTris=faceArray.length/3;
    for(var f=0;f<numTris;f++)
    {
        var fid=f*3;
        lineArray.push(faceArray[fid]);
        lineArray.push(faceArray[fid+1]);
        
        lineArray.push(faceArray[fid+1]);
        lineArray.push(faceArray[fid+2]);
        
        lineArray.push(faceArray[fid+2]);
        lineArray.push(faceArray[fid]);
    }
}

//-------------------------------------------------------------------------


