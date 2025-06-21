
/*
 * The Worker class allows for replacing a traditional loop with an asynchronous worker. 
 * It provides a way to run data-intensive tasks without blocking the main thread, 
 * enabling smooth UI updates and responsiveness.
 *
 * Usage:
 * 1. Create a new Worker instance with the following arguments:
 *    - chainlink: A reference to a chainlink object containing the inputControl.
 *    - task: An async function that will be executed in each iteration of the loop.
 *           It takes the iterator (i) as a parameter.
 *    - label: A string representing the label to be displayed in the progress bar.
 *    - length: The number of iterations the loop should run.
 *    - delay: The time (in milliseconds) to wait between each iteration.
 *
 * 2. Call the 'run()' method on the Worker instance. This method returns a Promise,
 *    which should be awaited to ensure the completion of the worker's task.
 *
 * Example:
 *
 * const worker = new MyWorker(chainlink, 
 *   async (i) => {
 *     // Write your loop logic here, using 'i' as the iterator
 *   },
 *   label,
 *   length,
 *   0
 * );
 *
 * await worker.run();
 */


const functionDefinitions = {
	
	
	shadowpoints: async (chainlink, dataset, yThreshold) => {
  
//////console.error(dataset);
  const worker = new MyWorker(chainlink,
    async (i) => {
  ////////console.warn(dataset.points.length*2-1);
		  ////////console.error(dataset.points[Math.floor(i/2)]);
      const point = dataset.points[Math.floor(i/2)].clone();

  	if(i % 2 == 0){
      return point;
}else{
      if (point.coordinates.y >= yThreshold) {
        return new Point({
          x: point.coordinates.x,
          y: parseFloat(yThreshold),
          z: point.coordinates.z,
          i: point.coordinates.i
        });
      }else{
		  return null;
	  }
      }
    },
    "Shadowing",
    dataset.points.length*2-1,
    0, Math.random()
  );

  await worker.run();
//////console.error(worker.result);
  await dataset.replacePoints(worker.result, chainlink);
  return dataset;
},


	polyhedron: async (chainlink, dataset, type, size) => {
  let points = [];
let i = 0;
  switch (type) {
    case 'tetrahedron':
      const t = Math.sqrt(2) * size;
      points.push(new Point({ x: -size, y: size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: -size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: size, z: size, i: i++ }));
      points.push(new Point({ x: -size, y: -size, z: size, i: i++ }));
      // outline of the first face
      points.push(new Point({ x: -size, y: size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: -size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: size, z: size, i: i++ }));
      points.push(new Point({ x: -size, y: size, z: -size, i: i++ }));
      // outline of the second face
      points.push(new Point({ x: -size, y: size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: size, z: size, i: i++ }));
      points.push(new Point({ x: -size, y: -size, z: size, i: i++ }));
      points.push(new Point({ x: -size, y: size, z: -size, i: i++ }));
      // outline of the third face
      points.push(new Point({ x: size, y: -size, z: -size, i: i++ }));
      points.push(new Point({ x: size, y: size, z: size, i: i++ }));
      points.push(new Point({ x: -size, y: -size, z: size, i: i++ }));
      points.push(new Point({ x: size, y: -size, z: -size, i: i++ }));
      // outline of the fourth face
      break;
case 'hexahedron':
  const h = size / 2;
  points.push(new Point({ x: -h, y: -h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: -h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: -h, i: i++ }));
  // outline of the first face
  points.push(new Point({ x: -h, y: -h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: -h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: -h, i: i++ }));
  // outline of the second face
  points.push(new Point({ x: -h, y: h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: h, z: -h, i: i++ }));
  // outline of the third face
  points.push(new Point({ x: -h, y: -h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: h, z: -h, i: i++ }));
  points.push(new Point({ x: h, y: h, z: -h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: -h, i: i++ }));
  // outline of the fourth face
  points.push(new Point({ x: -h, y: -h, z: h, i: i++ }));
  points.push(new Point({ x: -h, y: h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: h, z: h, i: i++ }));
  points.push(new Point({ x: h, y: -h, z: h, i: i++ }));
  // outline of the fifth face
  points.push(new Point({ x: -h, y: -h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: h, z: -h, i: i++ }));
  points.push(new Point({ x: -h, y: h, z: h, i: i++ }));
  points.push(new Point({ x: -h, y: -h, z: h, i: i++ }));
  // outline of the sixth face
  break;
case 'octahedron':
  const o = size / Math.sqrt(2);
  points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: -o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: o, i: i++ }));
  points.push(new Point({ x: o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
  // outline of the first face
  points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: -o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: o, i: i++ }));
  points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
  // outline of the second face
  points.push(new Point({ x: 0, y: -o, z: 0, i: i++ }));
  points.push(new Point({ x: o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: o, i: i++ }));
  points.push(new Point({ x: 0, y: -o, z: 0, i: i++ }));
  // outline of the third face
  points.push(new Point({ x: o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: o, i: i++ }));
  points.push(new Point({ x: o, y: 0, z: 0, i: i++ }));
  // outline of the fourth face
  points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
  points.push(new Point({ x: o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
  // outline of the fifth face
  points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
  points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
  // outline of the sixth face
  points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
points.push(new Point({ x: 0, y: o, z: 0, i: i++ }));
points.push(new Point({ x: -o, y: 0, z: 0, i: i++ }));
points.push(new Point({ x: 0, y: 0, z: -o, i: i++ }));
points.push(new Point({ x: 0, y: -o, z: 0, i: i++ }));
break;
case 'dodecahedron':
  const p = (1 + Math.sqrt(5)) / 2;
  const d = size / Math.sqrt(p);

  points.push(new Point({ x: d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: -d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: -d, y: d, z: -d, i: i++ }));
  points.push(new Point({ x: d, y: d, z: -d, i: i++ }));
  points.push(new Point({ x: d, y: -d, z: d, i: i++ }));
  points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));
  points.push(new Point({ x: -d, y: -d, z: -d, i: i++ }));
  points.push(new Point({ x: d, y: -d, z: -d, i: i++ }));

  // first pentagon
  points.push(new Point({ x: d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: -d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: -d * p, y: d / p, z: 0, i: i++ }));
  points.push(new Point({ x: 0, y: d / p, z: d * p, i: i++ }));
  points.push(new Point({ x: d * p, y: d / p, z: 0, i: i++ }));
  points.push(new Point({ x: d, y: d, z: d, i: i++ }));

  // second pentagon
  points.push(new Point({ x: d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: d * p, y: d / p, z: 0, i: i++ }));
  points.push(new Point({ x: d / p, y: 0, z: d * p, i: i++ }));
  points.push(new Point({ x: 0, y: d / p, z: d * p, i: i++ }));
  points.push(new Point({ x: d / p, y: 0, z: d * p, i: i++ }));
  points.push(new Point({ x: -d, y: d, z: d, i: i++ }));

  // third pentagon
  points.push(new Point({ x: -d, y: d, z: d, i: i++ }));
  points.push(new Point({ x: -d / p, y: 0, z: d * p, i: i++ }));
  points.push(new Point({ x: -d * p, y: d / p, z: 0, i: i++ }));
  points.push(new Point({ x: -d / p, y: 0, z: d * p, i: i++ }));
  points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
  points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));

  // fourth pentagon
  points.push(new Point({x: -d, y: -d, z: d, i: i++ }));
points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
points.push(new Point({ x: -d / p, y: 0, z: -d * p, i: i++ }));
points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
points.push(new Point({ x: -d / p, y: 0, z: d * p, i: i++ }));
points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));

// fifth pentagon
points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));
points.push(new Point({ x: d / p, y: 0, z: d * p, i: i++ }));
points.push(new Point({ x: 0, y: -d / p, z: d * p, i: i++ }));
points.push(new Point({ x: -d / p, y: 0, z: d * p, i: i++ }));
points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));

// first star
points.push(new Point({ x: d, y: d, z: d, i: i++ }));
points.push(new Point({ x: d * p, y: d / p, z: 0, i: i++ }));
points.push(new Point({ x: d / p, y: 0, z: d * p, i: i++ }));
points.push(new Point({ x: d / p, y: 0, z: -d * p, i: i++ }));
points.push(new Point({ x: d * p, y: d / p, z: 0, i: i++ }));
points.push(new Point({ x: d, y: d, z: -d, i: i++ }));
points.push(new Point({ x: 0, y: d * p, z: -d / p, i: i++ }));
points.push(new Point({ x: -d, y: d, z: -d, i: i++ }));
points.push(new Point({ x: -d, y: d, z: d, i: i++ }));
points.push(new Point({ x: 0, y: d * p, z: d / p, i: i++ }));
points.push(new Point({ x: d, y: d, z: d, i: i++ }));

// second star
points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));
points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
points.push(new Point({ x: -d / p, y: 0, z: d * p, i: i++ }));
points.push(new Point({ x: -d / p, y: 0, z: -d * p, i: i++ }));
points.push(new Point({ x: -d * p, y: -d / p, z: 0, i: i++ }));
points.push(new Point({ x: -d, y: -d, z: -d, i: i++ }));
points.push(new Point({ x: 0, y: -d * p, z: -d / p, i: i++ }));
points.push(new Point({ x: d, y: -d, z: -d, i: i++ }));
points.push(new Point({ x: d, y: -d, z: d, i: i++ }));
points.push(new Point({ x: 0, y: -d * p, z: d / p, i: i++ }));
points.push(new Point({ x: -d, y: -d, z: d, i: i++ }));
break;
case 'icosahedron':
const r = size * Math.sqrt(2 / (5 - Math.sqrt(5)));
 h = size * Math.sqrt((3 - Math.sqrt(5)) / 2);
points.push(new Point({ x: 0, y: r, z: -h }));
points.push(new Point({ x: r * 0.5, y: -h, z: 0.5 * size }));
points.push(new Point({ x: -r * 0.5, y: -h, z: 0.5 * size }));
points.push(new Point({ x: 0, y: r, z: h }));
points.push(new Point({ x: h, y: 0.5 * size, z: r * 0.5 }));
points.push(new Point({ x: -h, y: 0.5 * size, z: r * 0.5 }));
points.push(new Point({ x: h, y: -0.5 * size, z: -r * 0.5 }));
points.push(new Point({ x: -h, y: -0.5 * size, z: -r * 0.5 }));
points.push(new Point({ x: r * 0.5, y: h, z: -0.5 * size }));
points.push(new Point({ x: -r * 0.5, y: h, z: -0.5 * size }));
break;
default:
//////////console.error(`Invalid Platonic solid type: ${type}`);
}

if(dataset.points && dataset.points && dataset.points[0]){
points = [...points, dataset.points];
////console.warn(points, chainlink);
await dataset.replacePoints(points, chainlink);
return dataset;
}else{
	points = [...points];
	//console.error(points);
	let returnVal = new DataSet();
	await returnVal.replacePoints(points, chainlink);
	//console.error(points);
	return returnVal;
}

},


	loadfitfile: async (chainlink, dataset, file) => {
		if(!file){
			return dataset;
		}
  try {
	  const mainID = Math.random()+"-"+Math.random();
	  chainlink.progress(mainID, 0, "File Progress");
		  
	  let fitfile = new FitFile(file, chainlink );
	  chainlink.progress(mainID, 0.1, "File Progress");
	  
	  let data;
	  if(chainlink.persistentData.file != file){
		  //console.warn(JSON.stringify(file));
     data = await fitfile.parseFitFile(file);
     chainlink.persistentData.file = file;
    chainlink.persistentData.data = data;
    }else{
		data = chainlink.persistentData.data;
	}
	chainlink.progress(mainID, .2, "File Progress");
    let points = [];
let initialLat = 0;
let initialLong = 0;
let initialAlt = 0;
let initialTime = 0;
let initialCoords;
    const loopID = Math.random();
    let intervals = [];
    let prevTime= 0;
    
    originalData = data;
    data = data.filter((d) => d.type === "data" );

const fieldCounts = {}; // create an object to hold the field counts
const metadata = {}; // create an object to hold the metadata values

data.forEach((d) => {
  if (d.type === "data") {
    Object.keys(d.fields || {}).forEach((key) => {
      if (key !== 'units' && key !== 'byteArrays') {
        if (fitMetadataFields.includes(key)) {
          metadata[key] = d.fields[key];
        } else {
          if (!fieldCounts[key]) {
            fieldCounts[key] = 1;
          } else {
            fieldCounts[key]++;
          }
        }
      }
    });
  }
});

const representationMap = (axesCounts, length) => {
  // Create an array of objects containing the axis and count information
  const axisCountsArr = Object.entries(axesCounts).map(([axis, count]) => ({ axis, count }));

  // Sort the array in descending order based on the count
  axisCountsArr.sort((a, b) => b.count - a.count);

  // Create the HTML table
  let htmlTemplate = `<table>
                        <thead>
                          <tr>
                            <th>Total Data Length: ${length}</th>
                            <th>Count</th>
                            <th>Percentage</th>
                          </tr>
                        </thead>
                        <tbody>`;

  // Loop through the array of axis counts and add a row to the HTML table for each axis
  axisCountsArr.forEach(({ axis, count }) => {
    const percentage = ((count / length) * 100).toFixed(2);
    htmlTemplate += `<tr>
                        <td>${axis}</td>
                        <td>${count}</td>
                        <td>${percentage}%</td>
                      </tr>`;
  });

  // Close the HTML table
  htmlTemplate += `</tbody>
                  </table>`;

  return htmlTemplate;
};
const allAxes = Object.keys(fieldCounts).sort((a, b) => fieldCounts[b] - fieldCounts[a]); // create an array to hold the unique keys
    const promptReturn = await chainlink.inputControl.interruptingPrompt([true, true, true, representationMap(fieldCounts, data.length)], ['checkbox','checkbox','checkbox','custom'], ['Require Positional Data','Normalize','Interpolate','Representation'], 10000, {includeOutputMapping:true, outputMappingInput:allAxes}, null ); 
    const remapping = chainlink.mappings['specialMapping'] || "x:positionLat,y:altitude,z:positionLong,t:timestamp,u:power,v:speed,d:distance,a:heartRate,b:cadence,c:grade";
    console.log(promptReturn);
    const positionOnly = promptReturn[0]=='on';
    const normalize = promptReturn[1]=='on';
    const interpolate = promptReturn[2]=='on';
    console.log(remapping);
    let j = 0;
    
    for (let i = 0; i < data.length; i++){//=downsample) 
		chainlink.progress(loopID, parseFloat(i)/data.length, "Extracting Axes");
		chainlink.progress(mainID, .2+.8*i/data.length, "File Progress")
		
		  const mappedData = {};
  remapping.split(',').forEach((mapping) => {
    const [axis, field] = mapping.split(':');
    //console.warn(axis, field);
    if(normalize){

		let dataValue = data[i].fields[field] || null;
		if(field == 'timestamp'){
			if(!initialTime&& (!positionOnly || (positionOnly && (data[i].fields?.positionLat != undefined && data[i].fields?.positionLong != undefined && data[i].fields?.altitude != undefined)))){//&&data[i].fields?.timestamp>895165050    REMEMBER LATER TO LOOK AT METADATA
			  initialTime = data[i].fields.timestamp;
			 /// console.error(initialTime);
			  
		 	}
		 	dataValue = dataValue-initialTime;
		}else if(field == 'positionLong'){
			  
				  if(!initialLong){
					  initialLong = dataValue;
				}
				 if(!initialCoords && initialLat && initialLong){
				 	 initialCoords = new RelativeCoordinates(initialLat, initialLong);
				 }
				 if(!initialCoords){
					 dataValue = 0;
				 }else{
							let distances = 	initialCoords.getRelativeCoordinates(data[i].fields?.positionLat, data[i].fields.positionLong);
		 let [xDist, zDist] = [distances.x, distances.y];
					 dataValue = zDist;
					// console.error(zDist);
				 }
		}else if(field == 'positionLat'){
			
				  if(!initialLat){
					  initialLat = dataValue;
				 }
				 if(!initialCoords && initialLat && initialLong){
				 	 initialCoords = new RelativeCoordinates(initialLat, initialLong);
				 }
				 if(!initialCoords){
					 dataValue = 0;
				 }else{
							let distances = 	initialCoords.getRelativeCoordinates(data[i].fields.positionLat, data[i].fields?.positionLong);
		 let [xDist, zDist] = [distances.x, distances.y];
					 dataValue = xDist;
					// console.error(xDist);
				 }
		}else if(field == 'altitude'){
			if(!initialAlt){
				  initialAlt = dataValue;
			 }
			 dataValue = dataValue-initialAlt;
		}
		
		mappedData[axis] = dataValue;
	}else{
    	mappedData[axis] = data[i].fields[field] || null;
    }
  });
		//console.warn(mappedData);
    
    
      if (data[i].fields) {
        if ((data[i].fields?.altitude||0)+(data[i].fields?.positionLong||0)+(data[i].fields?.positionLat||0)+(data[i].fields?.timestamp||0)+(data[i].fields?.power||0)+(data[i].fields?.speed||0)+(data[i].fields?.distance||0)+(data[i].fields?.heartRate||0)+(data[i].fields?.cadence||0)+(data[i].fields?.grade||0)) {
         
const pointData = {};
Object.keys(mappedData).forEach((key) => {
  pointData[key.toLowerCase()] = mappedData[key] || null;
});

const mustHaveAllValues = false; // set this to true if every field must have a value to push a point, false if at least one field must have a value
const hasNonNullValue = Object.values(pointData).some((value) => value !== null);
if(!positionOnly || (positionOnly && (pointData.x || pointData.x==0)&& (pointData.y || pointData.y==0)&& (pointData.z || pointData.z==0))){
if (!mustHaveAllValues && hasNonNullValue || mustHaveAllValues && Object.values(pointData).every((value) => value !== null)) {
	//console.warn(pointData);
	pointData.i = j;
	j++;
  points.push(new Point(pointData));
}
}
         
         
          const prev = prevTime;
          if(data[i].fields.timestamp){
			 prevTime = data[i].fields.timestamp-initialTime;
		 }
          const current = data[i].fields.timestamp - initialTime;
          const next = data[i+1]?data[i+1].fields.timestamp - initialTime:0;
          const start = (prev + current) / parseFloat(120) ;
          const end = (next + current) / parseFloat(120);///2 and /60 for average and conversion to minutes
          const duration = end - start;
          const power = data[i].fields.power;
          if((start||start==0) && end && (power||power==0) && duration){
          	intervals.push(new Interval(start, end, new Point({x:duration, y:power})));
          }
        }
      }
    }
    
    chainlink.progress(loopID, -1, "Extracting Axes");
    chainlink.progress(mainID, 1, "File Progress");
//console.warn("end");
//console.warn(points);
dataset = new DataSet(points, intervals);
//console.warn(points);

await dataset.replacePoints(points, chainlink);
chainlink.progress(mainID, -1, "File Progress");
    return dataset;
  } catch (error) {
    return dataset;
  }
},
	cropaxis: async (chainlink, dataset, axis, cutoff, direction) => {
  const points = dataset.points.filter((point) => {
    const value = point.coordinates[axis];
    const range = dataset.getRange(axis);
    const rangeSize = range.max - range.min;
    const cutoffValue = direction === 'above' ? range.max - rangeSize * cutoff/100.0 : range.min + rangeSize * cutoff/100.0;
    return direction === 'above' ? value <= cutoffValue : value >= cutoffValue;
  });
  await dataset.replacePoints(points, chainlink);
  return dataset;
},
	createintervals: async (chainlink, dataset, minDuration, maxPrecision)=>{
		dataset.intervals = DataProcessor.getIntervalsFromPoints(dataset.points, dataset.interpolator);
		
		////////////////////////console.warn(dataset.intervals);
		return dataset;
	},
	powercurveconformer: (dataset, domainAxis, rangeAxis, controlPoints) => {


  return dataset;
},
		domainrepeater: async (chainlink, dataset, count, xAxis) => {
			const points = dataset.points;
			let returnPoints = [];
		domainLength = points[points.length-1] - points[0] || 1;
			for(let i = 0; i < count; i++){
				for(let j = 0; j < points.length; j++){
					let point = points[j].clone();
					point.coordinates[xAxis] +=  i * domainLength;
					returnPoints.push(point);
				}
			}////////////////////////////console.warn(returnPoints);
			await dataset.replacePoints(returnPoints, chainlink);
			return dataset;
		},
addpoints: async (chainlink, dataset, pointString) => {
	const addpoint = functionDefinitions.addpoint;
    let points = dataset.points;
    ////////////////console.log(points);
    const pointsData = pointString.split('|').filter(point => point !== '' );
    ////////////////console.log(pointsData);
    for(let i = 0; i < pointsData.length; i++){
        const pointData = pointsData[i];
        addpoint(chainlink, dataset, pointData);
    }
    return dataset;
},
addpoint: async (chainlink, dataset, pointString) => {
    let points = dataset.points;

    const pointData = pointString.split(',').filter(point => point !== '');
    const coordinates = {};

    for (let i = 0; i < pointData.length; i++) {
        const [label, value] = pointData[i].split(':');
        if(label == 'none' || label == ''){
            continue;
        }
        coordinates[label] = parseFloat(value);
    }

    const point = new Point(coordinates);

    if (Point.validPoint(point)) {
        ////////////////////////////console.warn("WE MADE A POINT", point);
        points.push(point);
        await dataset.replacePoints(points, chainlink);
    }

    return dataset;
},
		mapAxes: async (chainlink, dataset,  a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) => {
			let points = dataset.points;  
			const axisMap = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16];
			  const originalAxes = points[0].coordinates;
			  const newAxes = [...originalAxes];

			  // Map the first eight axes to the second eight axes
			  for (let i = 0; i < 8; i++) {
			    const sourceAxis = axisMap[i];
			    const destAxis = axisMap[i + 8];
			    if (sourceAxis !== 'none' && destAxis !== 'none') {
			      newAxes[destAxis] = originalAxes[sourceAxis];
			    }
			  }

			  // Extract the mapped axes
			  const mappedPoints = points.map(p => {
			    const mappedCoordinates = newAxes.reduce((acc, cur, idx) => {
			      if (cur !== 'none') {
			        acc[cur] = p.coordinates[idx];
			      }
			      return acc;
			    }, {});
			    return new Point(mappedCoordinates);
			  });
await dataset.replacePoints(points, chainlink);
			  return dataset;
			},
		sort: async (chainlink, dataset, firstAxis, firstDescending, secondAxis, secondDescending, thirdAxis, thirdDescending, asynchronous) => {
			  let points = [];
			  for(let i = 0; i < dataset.points.length; i++){
			    points.push(dataset.points[i].clone());
			  }

			  // Define a helper function to sort an array of values
			  const sortValues = (arr, desc) => {
			    const sorted = [...arr].sort();
			    if (desc) {
			      sorted.reverse();
			    }
			    return sorted;
			  };

			  // Sort each axis independently if asynchronous is true
			  if (asynchronous) {
			    const firstAxisValues = sortValues(points.map(p => p.coordinates[firstAxis]), firstDescending);
			    const secondAxisValues = sortValues(points.map(p => p.coordinates[secondAxis]), secondDescending);
			    const thirdAxisValues = sortValues(points.map(p => p.coordinates[thirdAxis]), thirdDescending);

			    points.forEach(p => {
			      p.coordinates[firstAxis] = firstAxisValues.shift();
			      p.coordinates[secondAxis] = secondAxisValues.shift();
			      p.coordinates[thirdAxis] = thirdAxisValues.shift();
			    });
			  } else {
			    // Sort points by the first axis
			    points.sort((a, b) => {
			      const aValue = a.coordinates[firstAxis];
			      const bValue = b.coordinates[firstAxis];
			      if (aValue > bValue) {
			        return firstDescending ? -1 : 1;
			      } else if (aValue < bValue) {
			        return firstDescending ? 1 : -1;
			      } else {
			        // If the first axis values are equal, sort by the second axis
			        const aSecondValue = a.coordinates[secondAxis];
			        const bSecondValue = b.coordinates[secondAxis];
			        if (aSecondValue > bSecondValue) {
			          return secondDescending ? -1 : 1;
			        } else if (aSecondValue < bSecondValue) {
			          return secondDescending ? 1 : -1;
			        } else {
			          // If the second axis values are equal, sort by the third axis
			          const aThirdValue = a.coordinates[thirdAxis];
			          const bThirdValue = b.coordinates[thirdAxis];
			          if (aThirdValue > bThirdValue) {
			            return thirdDescending ? -1 : 1;
			          } else if (aThirdValue < bThirdValue) {
			            return thirdDescending ? 1 : -1;
			          } else {
			            return 0;
			          }
			        }
			      }
			    });
			  }

			  await dataset.replacePoints(points, chainlink);
			  return dataset;
			},
				  perspective: async (
  chainlink,
  dataset,
  fixedPhi = true,
  fixedTheta = true,
  fixedPsi = true,
  phiValue = 0,
  thetaValue = 0,
  psiValue = 0,
  projectionType = "perspective"
) => {
  const originalPoints = dataset.points;
  const xVals = [];
  const yVals = [];
  const zVals = [];
  const phiVals = [];
  const thetaVals = [];
  const psiVals = [];

  // Extract relevant coordinates from each point
  originalPoints.forEach((point) => {
    xVals.push(point.coordinates.x);
    yVals.push(point.coordinates.y);
    zVals.push(point.coordinates.z ?? 0);
    phiVals.push(fixedPhi ? phiValue : point.coordinates.phi);
    thetaVals.push(fixedTheta ? thetaValue : point.coordinates.theta);
    psiVals.push(fixedPsi ? psiValue : point.coordinates.psi);
  });

  const rotatePoint = (x, y, z, phi, theta, psi) => {
    const cosPhi = Math.cos(phi);
    const sinPhi = Math.sin(phi);
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);
    const cosPsi = Math.cos(psi);
    const sinPsi = Math.sin(psi);

    const x1 = cosTheta * x + sinTheta * z;
    const y1 = y;
    const z1 = -sinTheta * x + cosTheta * z;

    const x2 = x1;
    const y2 = cosPhi * y1 - sinPhi * z1;
    const z2 = sinPhi * y1 + cosPhi * z1;

    const x3 = cosPsi * x2 - sinPsi * y2;
    const y3 = sinPsi * x2 + cosPsi * y2;
    const z3 = z2;

    return { x: x3, y: y3, z: z3 };
  };

  const advancedPerspective = (x, y, z, projectionType) => {
    let x2d, y2d, z2d;
    if (projectionType === "perspective") {
      const d = 10;
      x2d = (d / (d - z)) * x;
      y2d = (d / (d - z)) * y;
      z2d = z;
    } else if (projectionType === "ortho") {
      x2d = x;
      y2d = y;
      z2d = z;
    } else if (projectionType === "dof") {
      const dofValue = 500;
      const depth = z;
      const scale = depth / dofValue;

      x2d = x * scale;
      y2d = y * scale;
      z2d = z;
    }

    return { x: x2d, y: y2d, z: z2d };
  };

  const worker = new MyWorker(
    chainlink,
    async (i) => {
      const x = xVals[i];
      const y = yVals[i];
      const z = zVals[i];
      const phi = phiVals[i];
      const theta = thetaVals[i];
      const psi = psiVals[i];

      const rotatedPoint = rotatePoint(x, y, z, phi, theta, psi);

  const projectedPoint = advancedPerspective(
    rotatedPoint.x,
    rotatedPoint.y,
    rotatedPoint.z,
    projectionType
  );

  return new Point({
    x: projectedPoint.x,
    y: projectedPoint.y,
    z: projectedPoint.z,
    phi: parseFloat(phi),
    theta: parseFloat(theta),
    psi: parseFloat(psi),
  });
},
"Projection Worker",
dataset.points.length,
0
);

const returnPoints = await worker.run();
await dataset.replacePoints(returnPoints, chainlink);
return dataset;
},
	/*
Function name: compressor

Parameters:

dataset (object): A DataSet object containing an array of .points, containing Point objects, whose sole property is .coordinates, which can contain any key which is then an axis name.
compressorCurve (string): A name indicating an entry in a const array of function refs with different interpolators and control points.
center (number): A numerical value that offsets the signal from its middle.
threshold (number): A numeric floating point value expressing the percent of the maximum distance from the middle of the range plus the value of the center parameter.
clip (number): A numeric floating point value expressing the percent of the maximum distance from the middle of the range plus the value of the center parameter.
window (number): A percent of the whole, but not allowed to be calculated as smaller than two samples.
timeAxis (string): The key name of the axis to look for in DataSet's Point.coordinates, which will be considered inside the compressor as x.
signalAxis (string): The axis to look for in the input DataSet, but it will be considered inside the function as y.
attack (number): A value that determines how much to delay acting to tamp down y.
decay (number): A value that describes how many signals to take to go from full dampening to no dampening after sustain has expired.
sustain (number): A value that determines how long of a time change in signal value is required before the y value is allowed to not be dampened anymore.
knee (number): A value that describes the degree to which the evaluate() function should modify the curvature of the evaluate(x) functions response curve at low and high values of signal.
Description:
This function takes a DataSet object and uses it to apply dynamic compression to the signal data along the specified signalAxis. It applies a compression curve defined by the compressorCurve parameter to the input signal, with center, threshold, clip, attack, sustain, decay, and knee parameters modifying the compression behavior. The timeAxis parameter gives the key name of the axis to look for in DataSet's Point.coordinates, which will be considered inside the compressor as x. Each point that is considered for output will have it's range of signal values calculated over a calculatedWindow length of samples, and simple linear interpolation should be used to determine what is going on in the signal data when calculations occur between input data points. Helper functions must be defined within the nameless function that contains the code block.		
*/
compressor: async (chainlink, dataset, compressorCurve, center, threshold, ratio, clip, window, timeAxis, signalAxis, attack, decay, sustain, knee, wetDry) => {
	const getCompressorFunction = (windowSignals, curve, threshold, ratio, knee, envelope) => {
		const movingAverage = (signal, windowSize) => {
			const movingAvg = [];
			let sum = 0;
			for (let i = 0; i < windowSize; i++) sum += signal[i];
			movingAvg.push(sum / windowSize);
			for (let i = windowSize; i < signal.length; i++) {
				sum = sum - signal[i - windowSize] + signal[i];
				movingAvg.push(sum / windowSize);
			}
			return movingAvg;
		};

		const curveSet = {
			threeStageLinear: [[0,.333,.666,1],[0,.25,.5,.75],'bezier'],
			sinewave: [[0,.2,.4,.6,.8,1],[0,0.309,0.587,0.809,0.951,1],'linear'],
			none: [[0, .5, 1], [0, .5, 1], 'linear']
		};
		const newCurve = curveSet[curve];
		const interpolator = new Interpolator(newCurve[0], [newCurve[1]]);
		interpolator.setType(newCurve[2]);
		let complexCurve = [];
		for (let i = 0; i < 100; i++) {
			const x = i / 100;
			let y;
			if(x < threshold){
				y = x;
				}else{
					y = threshold + (interpolator.evaluate((x-threshold)/(1-threshold))/ratio)/(1-threshold);
				}
			//////////////////////////////console.log(x, y);
			complexCurve.push(y);
		}
		//interpolator.resetInterpolators()
		//complexCurve = movingAverage(complexCurve, Math.max(2, Math.round(knee * 10)));
		//////////////////////////////console.log(complexCurve);
		
			interpolator.resetInterpolators((new Array(100)).fill().map((_, i) => i / 99), [complexCurve]);
		
			const hardKnee = (x) => x < threshold ? x : threshold + (x - threshold) / knee;
		const softKnee = (x) => x < threshold ? x : threshold + (x - threshold) / (1 + (knee - 1) * Math.pow((x - threshold) / (1 - threshold), knee - 1));
		const compressorFunc = (x, range) => {
			let newSignal = interpolator.evaluate(x, {method: 'linear'});
			//////////////////////////////console.log(newSignal);
			return { signal: newSignal, ratio: x/newSignal };
		};
		return compressorFunc;
	};

	const calculateWeightedMean = (values) => {
		let sum = 0, weightSum = 0;
		for (let i = 0; i < values.length; i++) {
			const weight = 1 / (values.length - i);
			sum += weight * values[i];
			weightSum += weight;
		}
		return sum / weightSum;
	};

	const calculateRange = (values) => {
		const min = Math.min(...values);
		const max = Math.max(...values);
		return max - min;
	};

	const getEnvelope = (attack,sustain, decay, threshold, signal) => {
const envelope = signal < threshold ? 1 : sustain;
const timeSinceThreshold = Math.max(signal - threshold, 0);
const attackTime = Math.min(timeSinceThreshold / attack, 1);
let decayTime = Math.min((timeSinceThreshold - attack * sustain) / decay, 1);
if (decayTime < 0) decayTime = 0;
return envelope * (attackTime * (1 - sustain) + sustain - decayTime);
};
			const timeValues = dataset.points.map(point => point.coordinates[timeAxis]);
			const domain = calculateRange(timeValues);
			window = domain * window / 100;
			if (window < 2) window = 2;
			sustain = parseFloat(sustain);
			center = parseFloat(center);
			ratio = parseFloat(ratio);
			knee = parseFloat(knee);
			threshold = parseFloat(threshold);
			let oldRatio = 1;

			const processedPoints = dataset.points.map((point, i, points) => {
				const time = point.coordinates[timeAxis];
				let signal = point.coordinates[signalAxis];
				const windowStartTime = time - window / 2;
				const windowEndTime = time + window / 2;
				const windowPoints = points.filter((p) => {
					const t = p.coordinates[timeAxis];
					return t >= windowStartTime && t <= windowEndTime;
				});
				const windowSignals = windowPoints.map((p) => p.coordinates[signalAxis]);
				const weightedMean = calculateWeightedMean(windowSignals);
				const range = calculateRange(windowSignals);
				let signs = [];
				for (let i = 0; i < windowSignals.length; i++) {
					signs.push(Math.sign(windowSignals[i]-weightedMean));
					windowSignals[i] = Math.abs(windowSignals[i] - weightedMean);
				}
				const normalizedRange = calculateRange(windowSignals);
				for (let i = 0; i < windowSignals.length; i++) {
					windowSignals[i] = windowSignals[i] / normalizedRange;
					if (windowSignals[i] > 1) windowSignals[i] = 1;
				}
				signal = signal-weightedMean;
				const signalSign = Math.sign(signal);
				signal = Math.abs(signal) / normalizedRange;
				
				//const calculatedRatio = 1 + (ratio - 1) * Math.pow((signal - threshold) / (1 - threshold), knee);
				const calculatedClip = clip;
				const envelope = getEnvelope(attack, sustain, decay, threshold, oldRatio);
				const compressorFunc = getCompressorFunction(windowSignals, compressorCurve, threshold, ratio, knee, envelope);
				let compressedSignal;
				if (range != 0) {
					const compressorReturn =  compressorFunc(signal, range);
					[compressedSignal, oldRatio] = [compressorReturn.signal, compressorReturn.ratio];
				} else {
					compressedSignal = signal;
					oldRatio = 1;
				}
				
				if (compressedSignal > calculatedClip) compressedSignal = calculatedClip;
				compressedSignal = signalSign * (compressedSignal * normalizedRange) + weightedMean;
				
				let processedSignal = wetDry * compressedSignal + point.coordinates[signalAxis] * (1-wetDry);
				
				if(!processedSignal){
					//////////////////////////////console.trace();
					//////////////////////////////console.log(i, processedSignal);
					processedSignal = 0;
				}
				return new Point({
					[timeAxis]: time,
					[signalAxis]: processedSignal,
				});
			});
			await dataset.replacePoints(processedPoints);
			return dataset;			
},
		convertAxes: async (chainlink, dataset, interpretation, rotation) => {
			  let points = dataset.points;
			  let maxRadius = null;
			  
			  const toPolar = (x, y) => {
			    let r = Math.sqrt(Math.pow(x, 2) + Math.pow(y, 2));
			    let theta = Math.atan2(y, x);
			    return {r: r, theta: theta};
			  }

			  const toRectangular = (r, theta) => {
			    let x = r * Math.cos(theta);
			    let y = r * Math.sin(theta);
			    return {x: x, y: y};
			  }

			  const rotate = (point, theta) => {
				//  ////////////////////////////////////console.log("Point and theta: "+JSON.stringify(point)+":"+theta);
			    let rotatedTheta = parseFloat(point.theta) + parseFloat(theta);
			    return {r: point.r, theta: rotatedTheta};
			  }


			  let polarPoints = [];


			  for (let i = 0; i < points.length; i++) {
			    let point = points[i];
			    let coords;
			    
			    if (interpretation === 'polar to rectangular') {
			      		coords = point.coordinates;
			    } else if (interpretation === 'rectangular to polar') {
			      coords = {
			        r: Math.sqrt(Math.pow(point.coordinates.x, 2) + Math.pow(point.coordinates.y, 2)),
			        theta: Math.atan2(point.coordinates.y, point.coordinates.x)
			      };
			    } 

			    let rotatedCoords = rotate(coords, rotation);
			    let newPoint = {
			      coordinates: {
			        r: rotatedCoords.r,
			        theta: rotatedCoords.theta,
			      }
			    };
			    
			    if (interpretation === 'polar to rectangular') {
			    	    let rectCoords = toRectangular(rotatedCoords.r, rotatedCoords.theta);
			    	    newPoint.coordinates.x = rectCoords.x;
			    	    newPoint.coordinates.y = rectCoords.y;
			    	}
			    else if (interpretation === 'rectangular to polar') {
			    	      newPoint.coordinates.r = rotatedCoords.r;
			    	      newPoint.coordinates.theta = rotatedCoords.theta;
			    	}
			    
			    polarPoints.push(newPoint);
			    
			    if (coords.r > maxRadius) {
			      maxRadius = coords.r;
			    }
			  }
			//let returnVal = new DataSet();
			  //returnVal.replacePoints(polarPoints);
			  await dataset.replacePoints(polarPoints);
			  //return returnVal;
			 return dataset;
			},
			
			
		ramerdouglaspeucker: async (chainlink, dataset, ramer, epsilon) => {
			const ramerDouglasPeucker = (x, y, epsilon) => {
			    const n = x.length;
			    if (n < 3) {
			        return [x, y];
			    }

			    const keep = new Array(n).fill(false);
			    keep[0] = true;
			    keep[n - 1] = true;

			    const stack = [[0, n - 1]];
			    while (stack.length > 0) {
			        const [i, j] = stack.pop();
			        let dmax = 0;
			        let index = i;

			        for (let k = i + 1; k < j; k++) {
			            const d = Interpolator.perpendicularDistance(x, y, i, j, k);
			            if (d > dmax) {
			                dmax = d;
			                index = k;
			            }
			        }

			        if (dmax > epsilon) {
			            stack.push([i, index]);
			            stack.push([index, j]);
			        } else {
			            for (let k = i + 1; k < j; k++) {
			                keep[k] = true;
			            }
			        }
			    }

			    const newX = [];
			    const newY = [];
			    for (let i = 0; i < n; i++) {
			        if (keep[i]) {
			            newX.push(x[i]);
			            newY.push(y[i]);
			        }
			    }

			    return [newX, newY];
			}
		  const douglasPeucker = (x, y, epsilon) => {
			    const stack = [[0, x.length - 1]];
			    const keep = new Array(x.length).fill(false);
			    keep[0] = true;
			    keep[x.length - 1] = true;

			    while (stack.length > 0) {
			        const [i, j] = stack.pop();
			        let dmax = 0;
			        let index = i;

			        for (let k = i + 1; k < j; k++) {
			            const d = Interpolator.perpendicularDistance(x, y, i, j, k);
			            if (d > dmax) {
			                dmax = d;
			                index = k;
			            }
			        }

			        if (dmax > epsilon) {
			            stack.push([i, index]);
			            stack.push([index, j]);
			        } else {
			            for (let k = i + 1; k < j; k++) {
			                keep[k] = true;
			            }
			        }
			    }

			    const newX = [];
			    const newY = [];
			    for (let i = 0; i < x.length; i++) {
			        if (keep[i]) {
			            newX.push(x[i]);
			            newY.push(y[i]);
			        }
			    }

			    // Check that there are at least 3 points
			    if (newX.length < 3) {
			        while (newX.length < 3) {
			            // Duplicate the first or last point
			            if (newX.length === 0) {
			                newX.push(x[0]);
			                newY.push(y[0]);
			            } else if (newX.length === 1) {
			                newX.push(x[x.length - 1]);
			                newY.push(y[y.length - 1]);
			            } else {
			                // Interpolate a new point between the existing points
			                const midX = (newX[0] + newX[1]) / 2;
			                const midY = (newY[0] + newY[1]) / 2;
			                newX.splice(1, 0, midX);
			                newY.splice(1, 0, midY);
			            }
			        }
			    }

			    return [newX, newY];
			}
		    const perpendicularDistance = (x, y, i, j, k) => {
		        const x1 = x[i];
		        const y1 = y[i];
		        const x2 = x[j];
		        const y2 = y[j];
		        const xk = x[k];
		        const yk = y[k];
		        const numer = Math.abs((y2 - y1) * xk - (x2 - x1) * yk + x2 * y1 - y2 * x1);
		        const denom = Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));
		       // //////////////////////////////////////console.log("PERP DIST: "+ numer/denom);
		        return numer / denom;
		    }

		  epsilon = 100/parseFloat(epsilon);
		  let xVals = [], yVals = [], newX = [], newY = [], points = [];
		  pointProfiles = dataset.returnAxes(['x', 'y'], false);
		  for(let i = 0; i < dataset.points.length; i++){
			  xVals.push(dataset.points[i].coordinates.x);
			  yVals.push(dataset.points[i].coordinates.y);
		  }
			if(ramer){
				[newX, newY] = ramerDouglasPeucker(xVals, yVals, epsilon);
			}else{
				[newX, newY] = douglasPeucker(xVals, yVals, epsilon);
			}
			
			for(let i = 0; i < newX.length; i++){
				points.push(new Point({x:newX[i],y:newY[i]}));
				//////////////////////////////////////console.log(points[points.length-1]);
			}
	      await dataset.replacePoints(points);
	      //////////////////////////////////////console.log(dataset);
	      return dataset;
	  },
	  movingAverage: async (chainlink, dataset, windowSize, averageType) => {
	      const points = dataset.points.map((point, i, arr) => {
		        const x = point.coordinates.x;
		        const yArray = arr.slice(Math.max(i - windowSize + 1, 0), i + 1).map(p => p.coordinates.y);
		        const y = (averageType === 'root-mean-square') ? Math.sqrt(yArray.reduce((acc, val) => acc + val * val, 0) / yArray.length) : yArray.reduce((acc, val) => acc + val, 0) / yArray.length;
		        return new Point([x, y]);
		      });	      
	      await dataset.replacePoints(points);
      return dataset;
    },
		parametric1: async (chainlink, dataset, equation1, equation2, equation3) => {
		      const _evalEquation = (equation, xValues, yValues) => {
		          const uFunc = new Function('x, y', `with (Math) { return ${equation.replace(/u/g, 'x')}; }`);
		          return xValues.map((xValue, i) => uFunc(xValue, yValues[i]));
		        };

		        const points = dataset.points;
		        const xValues = points.map(point => point.coordinates.x);
		        const yValues = points.map(point => point.coordinates.y);
		        const uValues = _evalEquation(equation3, xValues, yValues);
		        const newPoints = uValues.map((uValue, i) => {
		          const newX = _evalEquation(equation1, [uValue], []);
		          const newY = _evalEquation(equation2, [uValue], []);
		        let returnVal = points[i].clone();
		        returnVal.x = newX;
		        returnVal.y = newY;
		        returnVal.u = uValue;
		          return returnVal;
		        });

		        await dataset.replacePoints(newPoints);
		        return dataset;
		      },
		modulator: async (chainlink, dataset, frequency, amplitude, phase, offset, timeAxis, modulationAxes, modulator) => {
			  const modulators = {
					    exp: Math.exp,
					    atan: Math.atan,
					    acos: Math.acos,
					    asin: Math.asin,
					    floor: Math.floor,
					    ceil: Math.ceil,
					    round: Math.round,
					    abs: Math.abs,
					    square: (x) => { return x * x; },
					    cube: (x) => { return x * x * x; },
					    sin: Math.sin,
					    cos: Math.cos,
					    tan: Math.tan,
					    log: Math.log,
					    log10: Math.log10,
					    random: Math.random,
					    lissajouscurve: (x, y, a, b, delta) => {
					      return Math.sin(a * x + delta) * Math.cos(b * y);
					    },
					    perlinNoise: (x, y) => {
					      const unit = 1 / Math.sqrt(2);
					      const vectors = [        [unit, unit],
					        [-unit, unit],
					        [unit, -unit],
					        [-unit, -unit],
					      ];
					      const gradients = vectors.map(vector => {
					        return [vector[0], vector[1], Math.random()];
					      });

					      const dx = x - Math.floor(x);
					      const dy = y - Math.floor(y);
					      const distances = [        [dx, dy],
					        [dx - 1, dy],
					        [dx, dy - 1],
					        [dx - 1, dy - 1],
					      ];
					      const dots = distances.map(distance => {
					        const gradient = gradients[Math.floor(x) % 4 + 2 * Math.floor(y) % 4];
					        return distance[0] * gradient[0] + distance[1] * gradient[1];
					      });

					      const u = 6 * dx * dx * dx * dx * dx - 15 * dx * dx * dx * dx + 10 * dx * dx * dx;
					      const v = 6 * dy * dy * dy * dy * dy - 15 * dy * dy * dy * dy + 10 * dy * dy * dy;
					      const result = dots[0] * (1 - u) * (1 - v) + dots[1] * u * (1 - v) + dots[2] * (1 - u) * v + dots[3] * u * v;

					      return result;
					    },
					    brownianMotion: (x, y, scale, octaves, persistence) => {
					      let amplitude = 1;
					      let frequency = 1;
					      let noise = 0;
					      for (let i = 0; i < octaves; i++) {
					        const perlin = modulators.perlinNoise(x * frequency, y * frequency);
					        noise += amplitude * perlin;
					        amplitude *= persistence;
					        frequency *= 2;
					      }
					      return noise * scale;
					      }
					    };

  const createNewPoints = async (point, timeAxis, modulationAxes, modulator, frequency, phase, amplitude, offset) => {
    const t = point.coordinates[timeAxis];
    const mod = modulators[modulator] ? modulators[modulator](parseFloat(t) * parseFloat(frequency) + parseFloat(phase), t * frequency + phase, 1, 2, 0) * amplitude + offset : 0;

    const newCoordinates = {};
    for (const axis of Object.keys(point.coordinates)) {
      let value = point.coordinates[axis];
      if (modulationAxes.includes(axis)) {
        value = value || 0;
        value += parseFloat(mod);
      }
      newCoordinates[axis] = parseFloat(value);
    }

    const newPoint = point.clone();
    newPoint.coordinates = newCoordinates;
    return newPoint;
  };

  const pointsWorker = new MyWorker(
    chainlink,
    async (i) => {
      const point = dataset.points[i];
      return await createNewPoints(point, timeAxis, modulationAxes, modulator, frequency, phase, amplitude, offset);
    },
    "Creating New Points",
    dataset.points.length,
    0
  );

  await pointsWorker.run();
  const newPoints = pointsWorker.result;

  await dataset.replacePoints(newPoints, chainlink);
  return dataset;
},
		maxMovingAveragePower: async (chainlink, dataset, density) => {
  const getWindowSizes = async (timeRange, density) => {
    const numPoints = Math.round(timeRange * density);

    const getWindowSizeWorker = new MyWorker(chainlink,
      async (i) => {
        const frac = i / numPoints;
        return frac * timeRange;
      },
      "Get Window Sizes",
      numPoints,
      0
    );

    await getWindowSizeWorker.run();
    return getWindowSizeWorker.result;
  };

  const getMaxPowers = async (signal, windowSizes, density) => {
    const maxPowers = [];
    const xs = [];

    const getMaxPowerWorker = new MyWorker(chainlink,
      async (i) => {
        const windowSize = windowSizes[i];
        const start = signal[0].coordinates.x;
        const end = start + windowSize;
        const subSignal = signal.filter(point => point.coordinates.x >= start && point.coordinates.x <= end);
        const subSignalValues = subSignal.map(point => point.coordinates.y);
        const movingAvg = movingAverage(subSignalValues, subSignalValues.length);
        const maxPower = Math.max(...movingAvg);
        return {maxPower, x: start + (end - start) / 2};
      },
      "Get Max Powers",
      windowSizes.length,
      0
    );

    await getMaxPowerWorker.run();
    getMaxPowerWorker.result.forEach(res => {
      maxPowers.push(res.maxPower);
      xs.push(res.x);
    });

    const densityFactor = Math.ceil(windowSizes.length / density);
    return {
      x: downsample(xs, densityFactor),
      y: downsample(maxPowers, densityFactor)
    };
  };

  const movingAverage = (signal, windowSize) => {
    const movingAvg = [];
    let sum = 0;
    for (let i = 0; i < windowSize; i++) {
      sum += signal[i];
    }
    movingAvg.push(sum / windowSize);
    for (let i = windowSize; i < signal.length; i++) {
      sum = sum - signal[i - windowSize] + signal[i];
      movingAvg.push(sum / windowSize);
    }
    return movingAvg;
  };

  const downsample = (array, factor) => {
    const downsampled = [];
    for (let i = 0; i < array.length; i += factor) {
      downsampled.push(array[i]);
    }
    return downsampled;
  };

  const signal = dataset.points.sort((a, b) => a.coordinates.x - b.coordinates.x);
  const timeRange = signal[signal.length - 1].coordinates.x - signal[0].coordinates.x;
  const windowSizes = await getWindowSizes(timeRange, density);
  const maxPowers = await getMaxPowers(signal, windowSizes, density);
  this.points = [];
const createPointsWorker = new MyWorker(chainlink,
    async (i) => {
      const point = new Point([maxPowers.x[i], maxPowers.y[i]]);
      return point;
    },
    "Create Points",
    maxPowers.x.length,
    0
  );

  await createPointsWorker.run();
  this.points = createPointsWorker.result;

  await dataset.replacePoints(this.points);
  return dataset;
},
			parametriccolorizer: async (chainlink, dataset, rEqn, gEqn, bEqn, aEqn, rKey, gKey, bKey, aKey, rBias, gBias, bBias, aBias)=> {
  const oldPoints = dataset.points.map(point => point.clone());

  const evaluateEquation = (eqn, value) => {
    let returnVal;
    try{
    	returnVal = eval(eqn.replace(/x/g, "("+value+")"));
    }catch{
		////console.warn(`I'm afraid that ${eqn.replace(/x/g, value)} cannot be evaluated: $returnVal`);
	}
    if (!returnVal && returnVal != 0) {
      returnVal = 0;
    }
    return returnVal;
  };

  const mapValue = (value, min, max) => {
    const returnVal = ((value - min) / (max - min));
    if (!returnVal) {
      return 0;
    }
    return returnVal;
  };

  const processWorker = new MyWorker(
    chainlink,
    async (i) => {
      const point = oldPoints[i];
      const rValue = rKey in point.coordinates ? evaluateEquation(rEqn, point.coordinates[rKey]) : null;
      const gValue = gKey in point.coordinates ? evaluateEquation(gEqn, point.coordinates[gKey]) : null;
      const bValue = bKey in point.coordinates ? evaluateEquation(bEqn, point.coordinates[bKey]) : null;
      const aValue = aKey in point.coordinates ? evaluateEquation(aEqn, point.coordinates[aKey]) : null;
      return { rValue, gValue, bValue, aValue };
    },
    "Processing Data",
    oldPoints.length,
    0
  );

  await processWorker.run();
  const values = processWorker.result;

  const rValues = values.map(item => item.rValue).filter(value => value !== null);
  const gValues = values.map(item => item.gValue).filter(value => value !== null);
  const bValues = values.map(item => item.bValue).filter(value => value !== null);
  const aValues = values.map(item => item.aValue).filter(value => value !== null);

  const minR = Math.min(...rValues);
  const minG = Math.min(...gValues);
  const minB = Math.min(...bValues);
  const minA = Math.min(...aValues);
  const maxR = Math.max(...rValues);
  const maxG = Math.max(...gValues);
  const maxB = Math.max(...bValues);
  const maxA = Math.max(...aValues);

  rBias = 2 - rBias;
  gBias = 2 - gBias;
  bBias = 2 - bBias;
  aBias = 2 - aBias;

  const newPointsWorker = new MyWorker(
    chainlink,
    async (i) => {
      const point = oldPoints[i];
      const { rValue, gValue, bValue, aValue } = values[i];
      point.coordinates.R = rValue !== null ? Math.round(Math.pow(mapValue(rValue, minR, maxR), rBias) * 255) : null;
      point.coordinates.G = gValue !== null ? Math.round(Math.pow(mapValue(gValue, minG, maxG), gBias) * 255) : null;
      point.coordinates.B = bValue !== null ? Math.round(Math.pow(mapValue(bValue, minB, maxB), bBias) * 255) : null;
      point.coordinates.A = aValue !== null ? Math.round(Math.pow(mapValue(aValue, minA, maxA), aBias) * 100+155) : null;
     return point;
    },
    "Generating New Points",
    oldPoints.length,
    0
  );

  await newPointsWorker.run();
  const newPoints = newPointsWorker.result;

  await dataset.replacePoints(newPoints, chainlink);
  return dataset;
},
		transformAxes: async (chainlink, dataset, xscale, yscale, xreverse, yreverse, xswap, yswap, xtranslate, ytranslate, rotate) => {
		    const points = dataset.points;
		    ////////////////////////////////////////console.log("POINTS: "+JSON.stringify(points));
		    let xmin = points[0].coordinates.x;
		    let xmax = points[0].coordinates.x;
		    let ymin = points[0].coordinates.y;
		    let ymax = points[0].coordinates.y;
		    let newPoints = [];
////////////////////////////////////console.error(xreverse);
		    // Calculate bounds
		    for (let i = 1; i < points.length; i++) {
		      const x = points[i].coordinates.x;
		      const y = points[i].coordinates.y;
		      if (x < xmin) {
		        xmin = x;
		      } else if (x > xmax) {
		        xmax = x;
		      }
		      if (y < ymin) {
		        ymin = y;
		      } else if (y > ymax) {
		        ymax = y;
		      }
		    }
//////////////////////////////////////console.log("before transforms: "+xmin+":"+xmax+"  "+ymin+":"+ymax);
		    // Apply transformations
		    for (let i = 0; i < points.length; i++) {
		      let newx = points[i].coordinates.x;
		      let newy = points[i].coordinates.y;

		      // Scale x and y
		      if (xscale !== 1) {
		        newx *= xscale;
		      }
		      if (yscale !== 1) {
		        newy *= yscale;
		      }

		      // Reverse x and/or y
		      if (xreverse) {
		        newx = -newx;
		      }
		      if (yreverse) {
		        newy = -newy;
		      }

		      // Swap x and y
		      if (xswap) {
		        const temp = newx;
		        newx = newy;
		        newy = temp;
		      }
		      if (yswap) {
		        const temp = newy;
		        newy = newx;
		        newx = temp;
		      }
////////////////////////////////////////console.log("Before translate: "+newx+":"+newy+"  "+xtranslate+":"+ytranslate);

		      // Translate x and y
		      
		      ////////////////////////////////////console.warn(newx +":"+ newy +":"+ xtranslate +":"+ ytranslate);
		      newx += parseFloat(xtranslate);
		      newy += parseFloat(ytranslate);
		      ////////////////////////////////////console.warn(newx +":"+ newy);
		      
		      // Rotate x and y
		      if (rotate !== 0) {
		        const angle = rotate * Math.PI / 180;
		        const relX = newx - (xmin + xmax) / 2;
		        const relY = newy - (ymin + ymax) / 2;
		        const newX = relX * Math.cos(angle) - relY * Math.sin(angle);
		        const newY = relX * Math.sin(angle) + relY * Math.cos(angle);
		        newx = newX + (xmin + xmax) / 2;
		        newy = newY + (ymin + ymax) / 2;
		      }
newPoints.push(points[i].clone());
		      newPoints[i].coordinates.x = newx;
		      newPoints[i].coordinates.y = newy;
		      //////////////////////////////////console.error(JSON.stringify(points[i]) + ":" + JSON.stringify(newPoints[i]));
		    }

		    await dataset.replacePoints(newPoints, chainlink);

		    return dataset;
		  },
			
		highLowFilter: async (chainlink, dataset, frequency, falloff, filterType) => {
		      const points = dataset.points.map(point => {
		    	  if(!this.prevX1&&!this.prevX2&&!this.prevY1&&!this.prevY2){
		    	  this.prevX1 = 0;
		  	    this.prevX2 = 0;
		  	    this.prevY1 = 0;
		  	    this.prevY2 = 0;
		    	  }
		    	  
			        const x = point.coordinates.x;
			        const y = point.coordinates.y;

			        // Calculate the filter coefficient
			        const omega = 2 * Math.PI * frequency;
			        const alpha = Math.sin(omega) / (2 * falloff);
			        const a0 = 1 + alpha;
			        const a1 = -2 * Math.cos(omega) / a0;
			        const a2 = (1 - alpha) / a0;
			        const b0 = (1 - Math.cos(omega)) / 2 / a0;
			        const b1 = (1 - Math.cos(omega)) / a0;
			        const b2 = (1 - Math.cos(omega)) / 2 / a0;
			        
			        // Apply the filter based on the chosen filter type
		const filteredY = (filterType === 'lowpass') ?
		    (b0 * y + b1 * this.prevY1 + b2 * this.prevY2 - a1 * this.prevX1 - a2 * this.prevX2) :
		    (b0 * y - b1 * this.prevY1 - b2 * this.prevY2 - a1 * this.prevX1 + a2 * this.prevX2);	        
			        // Update the previous input and output values
			        this.prevX2 = this.prevX1;
			        this.prevX1 = y;
			        this.prevY2 = this.prevY1;
			        this.prevY1 = filteredY;

			        return new Point([ x, filteredY ]);
			      });

			      await dataset.replacePoints(points);
			      return dataset;
			    },
	quantize: async (chainlink, dataset, domainAxis = 'x', rangeAxis = 'y', domainQuanta = 1, rangeQuanta = 1, outputAxis = 'z', errorAxis = 'e', maxError = .02, type = 'bellCurve') => {
  const errorResolver = (subsignal, maxError, errorAxis) => {
    //define errorResolver
    return subsignal;
  }
const quantizationFunction = (quantaVals, method, values) => {
  const methods = {
    min: (values) => {
      return values.map(val => {
        const idx = quantaVals.findIndex(quanta => val < quanta);
        if (idx > 0) {
          return parseFloat(quantaVals[idx - 1]);
        } else {
          return 0;
        }
      });
    },
    max: (values) => {
      return values.map(val => {
        const idx = quantaVals.findIndex(quanta => val < quanta);
        if (idx > 0 && idx < quantaVals.length) {
          return parseFloat(quantaVals[idx]);
        } else {
          return parseFloat(quantaVals[quantaVals.length - 1]);
        }
      });
    },
    round: (values) => {
      return values.map(val => {
        const idx = quantaVals.findIndex(quanta => val < quanta);
        if (idx > 0 && idx < quantaVals.length) {
          const deltaPrev = Math.abs(val - parseFloat(quantaVals[idx - 1]));
          const deltaNext = Math.abs(parseFloat(quantaVals[idx]) - val);
          if (deltaPrev < deltaNext) {
            return parseFloat(quantaVals[idx - 1]);
          } else {
            return parseFloat(quantaVals[idx]);
          }
        } else if (idx === 0) {
          return parseFloat(quantaVals[0]);
        } else {
          return parseFloat(quantaVals[quantaVals.length - 1]);
        }
      });
    },
    bellCurve: (values) => {
      const variance = Math.pow(Math.max(...values) - Math.min(...values), 2) / 16;
      return values.map(val => {
        const idx = quantaVals.findIndex(quanta => val < quanta);
        if (idx > 0 && idx < quantaVals.length) {
          const q1 = parseFloat(quantaVals[idx - 1]);
          const q2 = parseFloat(quantaVals[idx]);
          const midpoint = (q1 + q2) / 2;
          const weight = Math.exp(-Math.pow(val - midpoint, 2) / (2 * variance));
          return q1 * (1 - weight) + q2 * weight;
        } else if (idx === 0) {
          return parseFloat(quantaVals[0]);
        } else {
          return parseFloat(quantaVals[quantaVals.length - 1]);
        }
      });
    },
  };
  return methods[method](values);
};
  //if quanta[1] is an array, they are the values to quantize to in an ascending list.
  //if quanta[1] is a single value, it is the span between the quanta
  const rangeAxisValues = dataset.points.map(point => point.coordinates[rangeAxis]);
  const rangeQuantaValues = Array.isArray(rangeQuanta)
    ? rangeQuanta
    : Array.from(
        { length: Math.ceil(Math.max(...rangeAxisValues) / rangeQuanta) + 1 },
        (_, i) => i * rangeQuanta
      );
  const quantizedRangeValues = quantizationFunction(rangeQuantaValues, type, rangeAxisValues);
  
  // record resultant error on the appropriate axis as the range error times the domain error
  const domainAxisValues = dataset.points.map(point => point.coordinates[domainAxis]);
  const domainQuantaValues = Array.isArray(domainQuanta)
    ? domainQuanta
    : Array.from(
        { length: Math.ceil(Math.max(...domainAxisValues) / domainQuanta) + 1 },
        (_, i) => i * domainQuanta
      );
      ////////////////////////console.log(domainQuanta, domainQuantaValues);
  const quantizedDomainValues = quantizationFunction(domainQuantaValues, type, domainAxisValues);
  const domainErrorValues = quantizedDomainValues.map((val, idx) => {
    const prevVal = idx > 0 ? quantizedDomainValues[idx - 1] : null;
    const nextVal = idx < quantizedDomainValues.length - 1 ? quantizedDomainValues[idx + 1] : null;
    const domainError = Math.min(Math.abs(val - parseFloat(prevVal)), Math.abs(val - parseFloat(nextVal)));
    return [dataset.points[idx].coordinates[domainAxis], domainError];
  });
  const rangeErrorValues = dataset.points.map((point, idx) => {
    const rangeError = parseFloat(point.coordinates[rangeAxis]) - parseFloat(quantizedRangeValues[idx]);
    return [point.coordinates[rangeAxis], rangeError];
  });
const errorValues = domainErrorValues.map(([domainVal, domainError], idx) => {
  const rangeError = parseFloat(rangeErrorValues[idx][1]);
  const error = domainError * rangeError;
  return [domainVal, error];
});
  ////////////////////////console.log(errorValues, rangeErrorValues, domainErrorValues, domainAxisValues, quantizedDomainValues);
  // find regions of error exceeding the maxError percent
  const errorRegions = [];
  let currentRegion = null;
  for (let i = 0; i < errorValues.length; i++) {
    if (Math.abs(errorValues[i][1]) > maxError) {
      if (!currentRegion) {
        currentRegion = { startIndex: i };
      }
      currentRegion.endIndex = i;
    } else {
      if (currentRegion) {
        errorRegions.push(currentRegion);
        currentRegion = null;
      }
    }
  }
  if (currentRegion) {
    errorRegions.push(currentRegion);
  }
  
  // send such regions to errorResolver and replace them with the return
  const outputValues = dataset.points.map(point => point.coordinates[outputAxis]);
  for (let i = errorRegions.length - 1; i >= 0; i--) {
    const region = errorRegions[i];
    const subsignal =dataset.points.slice(region.startIndex, region.endIndex + 1);
			    
			    
	
const resolvedSubsignal = errorResolver(subsignal, maxError, errorAxis);
for (let j = region.startIndex; j <= region.endIndex; j++) {
  outputValues[j] = resolvedSubsignal[j - region.startIndex][outputAxis];
}
}

  // Update domain and range axes values to their quantized counterparts
  dataset.points.forEach((point, idx) => {
    point.coordinates[domainAxis] = quantizedDomainValues[idx];
    point.coordinates[rangeAxis] = quantizedRangeValues[idx];
  });
////////////////////////console.log(dataset.points);
  await dataset.replacePoints(dataset.points, chainlink);
  return dataset;

			    
			    },
		  interpolate: async (chainlink, dataset, domainAxis, rangeAxes,interpType, step, stepType, outputType) => {
			  //////////////////console.warn(rangeAxes);
			rangeAxes = rangeAxes.split('|').filter(axis => axis !== '' && axis !== 'none');
			if(!rangeAxes.length){
				return dataset;
			}
			 ////////////////console.warn(rangeAxes);
	    	      let points = dataset.points;
	    	        const domainValues = dataset.points.map((point) => point.coordinates[domainAxis]);
  const rangeValues = rangeAxes.map((rangeAxis) => dataset.points.map((point) => point.coordinates[rangeAxis]));
  ////////////////console.warn(rangeValues);
  const interpolator = new Interpolator(domainValues, rangeValues);
  interpolator.setType(interpType);
//  interpolator.resetInterpolators(domainValues, rangeValues);

	    	      let newPoints = [];
	    	      let numSteps = 0;
	    	      let stepSize = 0;
	    	      if (stepType === 'fractions') {
	    	        stepSize = (domainValues[domainValues.length - 1] - domainValues[0]) / parseFloat(step);
	    	      } else {
	    	        stepSize = 1 / parseFloat(step);
	    	      }
	    	      numSteps = Math.floor((domainValues[domainValues.length - 1] - domainValues[0]) / parseFloat(stepSize));
	    	      for (let i = 0; i <= numSteps; i++) {
	    	        let xi = domainValues[0] + i * stepSize;
	    	        let yi;
	    	        switch (outputType) {
	    	          case 'Integral':
	    	            yi = interpolator.integral(domainValues[0], xi);
	    	            ////////////////////////////////////////console.log("NEW INTEGRAL: "+xi+":"+yi);
	    	            break;
	    	          case 'Derivative':
	    	            yi = interpolator.derivative_at_x(xi);
	    	            ////////////////////////////////////////console.log("NEW DERIVATIVE: "+xi+":"+yi);
	    	            break;
	    	          default:
						  if(interpType == 'linear'){
	    	            yi = interpolator.evaluate(xi, {method: 'linear'});
	    	            }else{
							yi = interpolator.evaluate(xi);
						}
	    	          ////////////////console.log("NEW POINT: "+xi+":"+yi);
	    	            break;
	    	        }
	        let coordinates = { [domainAxis]: xi };
    if (Array.isArray(yi)) {
      for (let j = 0; j < yi.length; j++) {
        coordinates[rangeAxes[j]] = yi[j];
      }
    } else {
      coordinates[rangeAxes[0]] = yi;
    }
	    	        newPoints.push(new Point(coordinates));
	    	      }
	    	      dataset.interpolator = interpolator;
	    	      await dataset.replacePoints(newPoints, chainlink);
	    	      return dataset;
	    	    
		  },
laithe: async (chainlink, dataset, x, y, z, phi, theta, steps, relative) => {
  let transformedPoints = [];

  if (relative) {
    let currentPhi = 0;
    let currentTheta = 0;
    let currentX = 0;
    let currentY = 0;
    let currentZ = 0;
//console.warn(x, y, z, phi, theta, steps, relative);
    // Loop through each step of the transformation
    const workerTwo = new MyWorker(
		chainlink, 
		async (j) => {
			//console.log(j);

      // Update the current rotation angles and translation distances
      currentPhi += parseFloat(phi);
      currentTheta += parseFloat(theta);
      currentX += x;
      currentY += y;
      currentZ += z;

      // Create a worker to handle the transformation of each point
      const worker = new MyWorker(
        null,
        async (i) => {
			const point = points[i];
          const { x: initialX, y: initialY, z: initialZ } = point.coordinates;

          const rotatedX = parseFloat(initialX) * Math.cos(currentPhi) - initialZ * Math.sin(currentPhi);
          const rotatedY = parseFloat(initialY);
          const rotatedZ = parseFloat(initialX) * Math.sin(currentPhi) + initialZ * Math.cos(currentPhi);

          const rotatedY2 = parseFloat(rotatedY) * Math.cos(currentTheta) - parseFloat(rotatedZ) * Math.sin(currentTheta);
          const rotatedZ2 = parseFloat(rotatedY) * Math.sin(currentTheta) + parseFloat(rotatedZ) * Math.cos(currentTheta);

          const transformedPoint = new Point({ x: parseFloat(rotatedX) + parseFloat(currentX), y: parseFloat(rotatedY2) + parseFloat(currentY), z: parseFloat(rotatedZ2) + parseFloat(currentZ) });
////console.warn(transformedPoint);
          return transformedPoint;
        },
        "Transforming points",
        dataset.points.length,
        0,
       
      );
//console.warn(dataset.points.length);
const returnVal = worker.runThreaded({
        points: dataset.points, currentPhi, currentTheta, currentX, currentY, currentZ, standardAxes,
      }, ['Point']); 
      //console.warn(returnVal);
      return returnVal;

    }, "Building Steps", parseFloat(steps)+1, 0);
    //console.warn(workerTwo);
    transformedPoints = await workerTwo.run();
     //console.error(transformedPoints);
      combinedPoints = dataset.points.concat(transformedPoints.flat());
  } else {
    // Loop through each point in the dataset
    const worker = new MyWorker(
		chainlink,
		
		async (i) => {
      const point = points[i];
      const { x: initialX, y: initialY, z: initialZ } = point.coordinates;
      // Create a worker to handle the transformation of each point
      const workerTwo = new MyWorker(
        null,
        async (j) => {
          

            // Calculate the current rotation angles
            const currentPhi = (j * phi) / steps;
            const currentTheta = (j * theta) / steps;

            // Calculate the current x, y, and z coordinates based on the current step
            const currentX = initialX + x / steps * j;
            const currentY = initialY + y / steps * j;
            const currentZ = initialZ + z / steps * j;

            // Rotate the current x, y, and z coordinates
            const rotatedX = currentX * Math.cos(currentPhi) - currentZ * Math.sin(currentPhi);
            const rotatedY = currentY;
            const rotatedZ = currentX * Math.sin(currentPhi) + currentZ * Math.cos(currentPhi);

            const rotatedY2 = rotatedY * Math.cos(currentTheta) - rotatedZ * Math.sin(currentTheta);
			const rotatedZ2 = rotatedY * Math.sin(currentTheta) + rotatedZ * Math.cos(currentTheta);

	        // Create a new Point object with the transformed coordinates
    	    const transformedPoint = new Point({ x: rotatedX, y: rotatedY2, z: rotatedZ2 });

	        return transformedPoint;
   	   	},
    "Building Steps",
    parseFloat(steps) + 1,
    0,
    
  );
  let returnVal = workerTwo.run();
  //console.warn(returnVal);
  return returnVal;
  },
		"Tranforming Points",
		dataset.points.length,
		0,
	);

  transformedPoints = await worker.runThreaded({standardAxes, steps, phi, theta, x, y, z, "points":dataset.points}, ['Point', 'MyWorker']);
  //console.error(transformedPoints.flat());
  //combinedPoints = dataset.points.concat(transformedPoints.flat());
  // Collect the transformed points from the worker results
 
}


// Replace the original points with the transformed points
await dataset.replacePoints(transformedPoints.flat(), chainlink);

return dataset;
},
		  map: async (chainlink, dataset, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16) => {
			  const axisMap = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16];
			  
				  const axisInputMap = [a1, a2, a3, a4, a5, a6, a7, a8];
				  const axisOutputMap = [a9, a10, a11, a12, a13, a14, a15, a16];
				  const availableAxes = Object.keys(dataset.points[0].coordinates);
					let inputParameterIndices = [];
					let outputPoints = [];
				  const validAxes = availableAxes.filter(axis => axisInputMap.includes(axis));
				  for(let j = 0; j < dataset.points.length; j++){
						let outputPoint = dataset.points[j].clone();
						outputPoints.push(outputPoint);
					}
				  for(let i = 0; i < validAxes.length; i++){
						inputParameterIndices.push(axisMap.indexOf(validAxes[i]));
						
						for(let j = 0; j < dataset.points.length; j++){
							
							outputPoints[j].coordinates[axisMap[inputParameterIndices[i]+8]] = dataset.points[j].coordinates[validAxes[i]];
						}
						
					}
				  

				  await dataset.replacePoints(outputPoints);
				  return dataset;
				},
}
		const standardAxes = ['x', 'y', 'z', 'r', 'phi', 'theta', 'psi', 'lamda', 'e', 'epsilon', 'mu', 'sigma', 'i', 'j', 'k', 't', 'd', 'u', 'v', 'a', 'b', 'c', 'R', 'G', 'B', 'A', 'none' ];
		
const availableFunctions = [

	{
		name: 'Load FIT File',
		description: 'load a FIT file in as data points',
		className: 'LoadFitFile',
		forge: {
			fn: 'loadfitfile',
			paramNames: ['file'],
			inputTypes: ['fileSelector'],
			labels: ['File'],
			inputParams:["fitfiles", ]
		}
		
	},
	{
		name: 'Shadow',
		description: 'put a shadow on the present data',
		className: 'Shadow',
		forge:{
			fn: 'shadowpoints',
			paramNames: ['yThreshold'],
			inputTypes: ['range'],
			labels:		['Floor'],
			inputParams:[[-100,100,1,0]]
		}
	},
		{
		name: 'Polyhedron',
		description: 'create a platonic solid',
		className: 'Polyhedron',
		forge: {
			fn: 'polyhedron',
			paramNames: ['type','size'],
			inputTypes: ['select', 'range'],
			labels: ['Type', 'Size'],
			inputParams: [[['tetrahedron', 'hexahedron', 'octahedron', 'dodecahedron', 'icosahedron'],'icosahedron'],[.1,100,.1,1]]
		}
	},
{
	name: 'Laithe',
	description: 'builds up points over rotation and translation, needs to be renamed',
	className: 'laithe',
	forge: {
		fn: 'laithe',
		paramNames: ['x','y','z','phi','theta', 'steps', 'relative'],
		inputTypes: ['range','range','range','range','range','range', 'checkbox'],
		labels:		["Translate X", "Translate Y", "Translate Z", "Rotate Phi", "Rotate Theta", "Steps", "Relative?"],
		inputParams:[[-100, 100, .1, 0],[-100, 100, .1, 0],[-100, 100, .1, 0],[-5*Math.PI, 5*Math.PI, .01, 0],[-5*Math.PI, 5*Math.PI, .01, 0],[1, 100, 1, 1], false]
	}
},
{
	
	name: 'Crop Axis',
	description: 'removes any point with an axis value outside the percent of range specified.',
	className: 'cropaxis',
	forge: {
		paramNames: ['axis', 'cutoff', 'direction'],
		inputTypes: ['axisSelect', 'range', 'select'],
		inputParams:['x', [0,100,.1,0], [['before','after'],'before']],
		labels:		['Axis', 'Cutoff Percent', 'Cut Before or After'],
		fn: 'cropaxis',
	}
},
{
	name : 'Quantize',
	description: 'Quantize range and domain values',
	className: "quantize",
	forge: {
		paramNames: ['domainAxis','rangeAxis','domainQuanta', 'rangeQuanta', 'outputAxis', 'errorAxis', 'maxError', 'type'],
		inputTypes: ['axisSelect', 'axisSelect', 'range', 'range', 'axisSelect', 'axisSelect', 'range', 'select'],
		inputParams:['x', 'y', [.001, 10, .001, 1], [.001, 10, .001, 1], 'y', 'e', [.001, .1, .001, .02], [['min', 'max', 'round', 'bellCurve'], 'round']],
		labels: 	['Domain Axis', 'Range Axis', 'Domain Quanta', 'Range Quanta', 'Output Axis', 'Error Axis', 'Maxmim Error Percent', 'Quantization Type'],
		fn: 'quantize'
	}
},
	{ name: 'Interval Creator',
		description: 'interpret a signal as a set of intervals',
		className: 'intervalCreator',
		forge:{
			paramNames: ['minDuration', 'maxPrecision'],
			inputTypes: ['range', 'range'],
			inputParams: [[.25, 5, .25, 1], [.25, 5, .25, 1]],
			labels: ['Minimum Duration', 'Minimum Fraction'],
			fn: 'createintervals',
		}
	},
	{
		name: 'Power Curve Conformer',
		description: 'Supply points defining power and duration desired, and the input signal will be transformed to try to reach those control points',
		className: "we don't use class names anymore do we?",
		forge:{
			paramNames: ['domainAxis', 'rangeAxis', 'controlPoints'],
			inputTypes: ['axisSelect', 'axisSelect', 'pointsSelect'],
			inputParams: ['x', 'y', [6,2]],
			labels: ['Domain', 'Range', 'Targets'],
			fn: 'powercurveconformer',
		}
		
		},

	{
		name: 'Domain Repeater',
		description: 'repeat values by repeating the domain n times', 
		className: 'domainRepeater',
		forge:{
			paramNames: ['count', 'xAxis'],
			inputTypes: ['range', 'axisSelect' ],
			inputParams: [[1,100,1,1], 'x'],
			labels: ['Count', 'Domain Axis'],
			fn: 'domainrepeater',
		}
		
	},
	{
		name: 'Add Points',
		description: 'add new points of your description', 
		className: 'insertPoints',
		forge:{
			paramNames: ['pointString'],
			inputTypes: ['pointsSelect'],
			inputParams: [[6,6]],
			labels: ['Points'],
			fn: 'addpoints',
		}
		
	},
	{
		name: 'Add A Point',
		description: 'add a point of your description', 
		className: 'insertPoint',
		forge:{
			paramNames: ['pointString'],
			inputTypes: ['pointSelect'],
			inputParams: [6],
			labels: ['Point'],
			fn: 'addpoint',
		}
		
	},
	{
		name: 'Map',
		description: 'remap axes',
		className: 'Map',
		forge:{
			paramNames: [ 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16' ],
			inputTypes: ['axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect' ],
			inputParams: ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none' ],
			labels: ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8' ],
			fn: 'map',
		}
	},
	{
		name: 'Sort',
		description: 'sort points by an axis',
		className: 'Sort',
		forge:{
			paramNames: ['firstAxis', 'firstDescending', 'secondAxis', 'secondDescending',  'thirdAxis', 'thirdDescending', 'asynchronous'],
			inputTypes: ['axisSelect', 'checkbox', 'axisSelect', 'checkbox', 'axisSelect', 'checkbox', 'checkbox' ],
			inputParams: ['x', false, 'none', false, 'none', false, false],
			labels: ['First Sort Axis', 'Descending', 'Second Sort Axis', 'Descending', 'Third Sort Axis', 'Descending', 'Asychronous' ],
			fn: 'sort',
		}
	},
	{ 
		name: 'Projector',
	    description: 'create a 2d projection from 3d(7d) data (point(3d), rotation angles(2d), camera angles(2d))',
	    className: 'Projector',
		forge:{
			paramNames: ['fixedPhi', 'fixedTheta', 'fixedPsi', 'phiValue', 'thetaValue', 'psiValue', 'projectionType'],
			inputTypes: ['checkbox', 'checkbox', 'checkbox', 'range', 'range', 'range', 'select'], 
			inputParams: [true, true, true, [-Math.PI, Math.PI, Math.PI/32, 0], [-Math.PI, Math.PI, Math.PI/32, 0], [-Math.PI, Math.PI, Math.PI/32, 0], [['perspective','ortho', 'dof', 'xz', 'xy', 'yz' ], 'perspective']], 
			labels: ['Constant Phi', 'Constant Theta', 'Constant Psi', 'Phi Value', 'Theta Value', 'Psi Value', 'Projection Style'],
			fn: 'perspective',
		}
},
	{
	name: 'Parametric Colorizer',
	description: 'Describe equations for R, G, B, and A, based on data axes',
	className: 'ParametricColorizer',
	forge: {
		paramNames: ['rEqn', 'gEqn', 'bEqn', 'aEqn', 'rKey', 'gKey', 'bKey', 'aKey', 'rBias', 'gBias', 'bBias', 'aBias'],
		inputTypes: ['text', 'text', 'text', 'text', 'axisSelect', 'axisSelect', 'axisSelect', 'axisSelect', 'range', 'range', 'range', 'range'],
		labels:		['Red(x)', 'Green(x)', 'Blue(x)', 'Alpha(x)', 'Red x', 'Green x', 'Blue x', 'Alpha x', 'Red Bias', 'Green Bias', 'Blue Bias', 'Alpha Bias'],
		fn: 'parametriccolorizer',
		inputParams:['x','x','x','x', 'x', 'y', 'z', 'i',[0, 2, .01, 1],[0, 2, .01, 1],[0, 2, .01, 1],[0, 2, .01, 1] ],	
	
		}


	},
	{
	name: 'Modulator',
	description: 'Add a function controlled by one axis to up to three axes',        
	className: 'ModulatorLink',        
	forge: {          
		paramNames: ['frequency', 'amplitude', 'phase', 'offset', 'timeAxis', 'modulationAxes', 'modulator'],
      inputTypes: ['range', 'range', 'range', 'range', 'axisSelect', 'checkboxgroup', 'select'],
      labels: ['Frequency', 'Amplitude', 'Phase', 'Offset', 'Time Axis', 'Modulation Axes', 'Modulator'],
      inputParams: [[0, 100, 1, 1], [0, 100, 1, 10], [-100, 100, 1, 0], [-120, 120, 1, 0], 'x', [['x', 'y', 'z', 'r', 'phi', 'theta', 'R', 'G', 'B', 'A'],'x'], [['sin', 'cos', 'tan', 'log', 'log10', 'asin', 'acos', 'atan', 'ceil', 'round', 'floor', 'abs', 'square', 'cube'], 'sin']],
      fn: 'modulator',
    },
  },
  {
    name: 'Hi/Lo Filter',
    description: 'A highpass or lowpass filter',
    className: 'FilterLink',
    forge: {
      paramNames: ['frequency', 'falloff', 'hilo'],
      inputTypes: ['range', 'range', 'radio' ],
      labels: ['Frequency', 'Falloff', 'Filter Type'],
      inputParams: [[0, 10000, 10, 1000], [0, 100000, 1, 1], [['lowpass', 'highpass'], 'lowpass']],
      fn: 'highLowFilter',
	  },
  },
  {
    name: 'Interpolator',
    description: 'Resample points from an interpolation curve',
    className: 'InterpolatorLink',
    forge: {
      fn: "interpolate",
	  paramNames:  ['domainAxis', 'rangeAxes','interpType',  'step', 'stepType', 'outputType'],
	   	  inputTypes:  ['axisSelect', 'axesSelect','select',  'range', 'checkbox', 'radio'],
	      labels:      ['Domain', 'Ranges', 'Algorithm', 'Density', 'Equal Spaces', 'Output'],
	      inputParams:['i', ['x', 'y', 'z', 'R', 'G', 'B', 'A'], [['linear', 'pchip', 'wavelet', 'catmull-rom', 'akima', 'poly', 'poly3', 'poly5', 'poly10', 'bspline', 'cubic', 'bezier'], 'linear'], [1, 100, 1, 10], true, [['Integral', 'Evaluate', 'Derivative'], 'Evaluate']],
    },
  },
  {
	    name: 'Moving Average',
	    description: 'Calculate a moving average',
	    className: 'MovingAverageLink',
	    forge: {
	        paramNames: ['windowSize', 'averageType'],
	        inputTypes: ['range', 'radio'],
	        labels: ['Window Size', 'Average Type'],
	        inputParams: [[2, 100, 1, 5], [['mean', 'root-mean-square'], 'mean']],
	        fn: 'movingAverage',
	    },
	},
	{
	    name: 'Max Moving Average',
	    description: 'Calculate a maximum moving average over window size curve',
	    className: 'MaxMovingAvgPowerLink',
	    forge: {
	        paramNames: ['density'],
	        inputTypes: ['range'],
	        labels: ['Density'],
	        inputParams: [[1, 1000, 1, 10]],
	        fn: 'maxMovingAveragePower',
	    },
	},
	{
		name: 'Axes Conversion',
		description: 'Remap, convert, colorize',
		forge: {
			fn: "convertAxes",
			paramNames: [ 'interpretation', 'rotation'],
		   inputTypes: [ 'radio', 'range'],
		   labels: [ 'Interpretation', 'Rotation'],
		   inputParams: [[['rectangular to polar', 'polar to rectangular', 'passthrough'], 'rectangular to polar'], [-2*Math.PI, 2*Math.PI, .01, 0]]
		}
	},
  {
    name: 'Axes Transformation',
    description: 'Transform the data points',
    className: 'AxesTransformationLink',
    forge: {
    	
    	    fn: "transformAxes",		        	    
    	    paramNames: ['xscale', 'yscale', 'xreverse', 'yreverse', 'xswap', 'yswap', 'xtranslate', 'ytranslate', 'rotate'],
    	    inputTypes: ['range', 'range', 'checkbox', 'checkbox', 'checkbox', 'checkbox', 'range', 'range', 'range'],
    	    labels: ['X Scale', 'Y Scale', 'X Reverse', 'Y Reverse', 'X Swap', 'Y Swap', 'X Translate', 'Y Translate', 'Rotate'],
    	    inputParams: [
    	      [0.01, 100, 0.1, 1], 
    	      [0.01, 100, 0.1, 1], 
    	      false, 
    	      false, 
    	      false, 
    	      false, 
    	      [-10, 10, 0.01, 0], 
    	      [-10, 10, 0.01, 0], 
    	      [-180, 180, 1, 0]
    	    ]
    	  
    	
    },
  },
  {
	name: 'Crop/Compress Range',
	description: 'a basic compressor and hard limiter ',
	className: 'CropCompressRange',
	forge: {
		fn: 'compressor',
		paramNames: ['compressorCurve', 'center', 'threshold', 'ratio', 'clip', 'window', 'timeAxis', 'signalAxis', 'attack', 'decay', 'sustain', 'knee', 'wetDry'],
		inputTypes: ['select', 'range', 'range', 'range', 'range', 'range', 'select', 'select', 'range', 'range', 'range', 'range', 'range'],
		labels:		['Compression', "Offset Center From Mean", "Compression Threshold", "Compression Ratio", "Clipping Threshold", "Window", "dx Axis", "dy Axis", "Attack", "Decay", "Sustain", "Knee", "Wet/Dry ratio"],
		inputParams: [
			[['none', 'sinewave', 'threeStageLinear'], 'logarithmic'], 
			[-100, 100, .1, 0], [0, 1, .01, 0], [.01, 15, .01, 1], [.1, 1, .01, 1], [2, 100, .1, 100],
			[['x', 'y', 'z','u','theta', 'r', 'R', 'G', 'B'], 'x'] , [['x', 'y', 'z','u','theta', 'r', 'R', 'G', 'B'], 'y'], 
			[.01, 1, .01, .1], [.01, 1, .01, .1], [.01, 1, .01, .1], [0, 1, .01, 0], [ 0.001, 1, .01, 0],
		]
	}
  },
  {
    name: 'Parametric Equation',
    description: 'define the u range from x and y and x and y from the u range',
    className: 'ParametricEquationLink',
    forge: {
      paramNames: ['equation1', 'equation2', 'equation3'],
      inputTypes:  ['text', 'text', 'text'],
      labels:  ['X(u)', 'Y(u)', 'U(x,y)'],
      inputParams: ["u^2+1","Math.sin(u)^2", "(x^2+y^2)^.5"],
      fn: 'parametric1',
    },
  },
  {
    name: 'Ramer-Douglas-Peucker',
    description: 'scrub outliers',
    className: 'DouglasPeuckerLink',
    forge: {
      paramNames: ['ramer', 'epsilon'],
      inputTypes: ['checkbox', 'range'],
      labels: ['Ramer', 'Epsilon'],
      inputParams: [ true, [1, 10000, 100, 1] ],
      fn: 'ramerdouglaspeucker',
    },
  },
];