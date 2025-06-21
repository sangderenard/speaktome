class ProgressBarManager {
  constructor() {
    this.progressBarMap = new Map();
    this.idCounter = 0;
  }

  getID() {
    return this.idCounter++;
  }
static AnimateProgressBar(container, onComplete) {
  // Set transition and width properties to animate progress bar to 100%
  container.style.transition = 'width 0.25s ease-in-out';
  container.style.width = '100%';

  // Call onComplete function after animation is complete
  setTimeout(() => {
    onComplete();
  }, 250);
}
  createProgressBar(inputControl) {
    // Create the container element
    const container = document.createElement('div');
    container.style.width = '300px';
    container.style.border = '1px solid black';
    container.style.position = 'relative';
    container.style.overflow = 'hidden';
inputControl.processID = this.getID();
    inputControl.progress = async (id, progress, label, killFunc) => {
		////////console.warn(killFunc);
      setTimeout(() => {
        let progressBar = this.progressBarMap.get(id);

if (progress === -1) {
  if (progressBar) {
    // Animate progress bar to 100% and remove it asynchronously
    setTimeout(() => {
      ProgressBarManager.AnimateProgressBar(progressBar.container, () => {
        if (container.contains(progressBar.container)) {
          container.removeChild(progressBar.container);
        }
        this.progressBarMap.delete(id);
      });
    }, 0);
  }
  return;
}

        if (!progressBar) {
        const barContainer = document.createElement('div');
        barContainer.style.width = '100%';
        barContainer.style.height = '30px';
        barContainer.style.position = 'relative';

        const bar = document.createElement('div');
        bar.style.height = '100%';
        bar.style.backgroundColor = 'blue';
        bar.style.position = 'absolute';

     const labelText = document.createElement('span');
labelText.style.position = 'absolute';
labelText.style.left = '50%';
labelText.style.top = '50%';
labelText.style.transform = 'translate(-50%, -50%)';
labelText.style.color = 'white';
labelText.style.fontWeight = 'bold';
labelText.style.textShadow = '0px 0px 2px black'; // Add text outline

        barContainer.appendChild(bar);
        barContainer.appendChild(labelText);

        progressBar = { container: barContainer, bar, labelText, killFunc };
        this.progressBarMap.set(id, progressBar);
      const closeButton = document.createElement('button');
          closeButton.textContent = 'x';
          closeButton.style.position = 'absolute';
          closeButton.style.right = '0';
          closeButton.style.top = '0';
          closeButton.style.border = 'none';
          closeButton.style.background = 'none';
          closeButton.style.color = 'white';
          closeButton.style.fontWeight = 'bold';
          closeButton.style.cursor = 'pointer';
          closeButton.onclick = () => {
            if (progressBar.killFunc) {
              progressBar.killFunc();
            }
            container.removeChild(progressBar.container);
            this.progressBarMap.delete(id);
          };

          barContainer.appendChild(closeButton);

          progressBar = { container: barContainer, bar, labelText, killFunc };
          this.progressBarMap.set(id, progressBar);
          ////////console.warn(id);
          let processPrefix, processSuffix;
           try{
           		[processPrefix, processSuffix] = id.split('-');
           }catch{
			   //////console.error("INVALID ID");
		   }
          for (const [key, otherProgressBar] of this.progressBarMap) {
            let otherPrefix, otherSuffix;
            try{
				[otherPrefix, otherSuffix] = key.split('-');
			}catch{
				//////console.error("INVALID ID");
			}
            if (processPrefix === otherPrefix && processSuffix !== otherSuffix) {
              ////////console.warn("MATCH FOUND");
              if(processSuffix < otherSuffix){
				  killFunc();
				  
				  this.progressBarMap.delete(id);
				  return;
			  }else{
              if (otherProgressBar.killFunc) {
				  ////////console.warn("killing old function");
                otherProgressBar.killFunc();
              }
              container.removeChild(otherProgressBar.container);
              this.progressBarMap.delete(key);
              }
            }
          }
          
        }
const percentage = Math.max(0, Math.min(100, progress * 100));
      progressBar.bar.style.width = `${percentage}%`;
      progressBar.labelText.textContent = label;

      // Remove the progress bar from the container to reposition it
      if (container.contains(progressBar.container)) {
        container.removeChild(progressBar.container);
      }

      // Insert the progress bar in the correct position based on the progress
      const sortedKeys = Array.from(this.progressBarMap.keys()).sort((a, b) => {
        return this.progressBarMap.get(b).bar.style.width.slice(0, -1) - this.progressBarMap.get(a).bar.style.width.slice(0, -1);
      });

      const index = sortedKeys.indexOf(id);
      if (index === container.childElementCount) {
        container.appendChild(progressBar.container);
      } else {
        container.insertBefore(progressBar.container, container.children[index]);
      }

      // Adjust the container height
      container.style.height = `${30 * this.progressBarMap.size}px`;
  
      }, 0);
    };

    return container;
  }
}

class MyWorker {
  constructor(chainlink, task, label, loopLimit, delay, parentID, numWorkers = 8) {
	  const dummyFunction = (...args) => {
  // Do nothing
};
    this.chainlink = chainlink;
    this.task = task;
   this.progressFunction = chainlink && chainlink.inputControl.progress || dummyFunction;

    this.label = label;
    this.loopLimit = loopLimit;
    this.delay = delay;
    this.currentIteration = 0;
    this.isRunning = false;
    this.isPaused = false;
    this.parentID = parentID;
    this.id = chainlink
      ? (parentID ? parentID + "+" : "") + chainlink.processID + "-" + chainlink.inputControl.progressBarManager.getID()
      : null;
    this.result = [];
    this.numWorkers = numWorkers;
  }

async run(iteratorCallback = null) {
  return new Promise((resolve, reject) => {
    this.isRunning = true;
    const skipIterations = Math.ceil(this.loopLimit * 0.01);
////////console.warn(this.loopLimit, skipIterations);
    const loop = async (i) => {
      try {
		//console.warn("LOOPING, ",i," of ", this.loopLimit);
        if (i < this.loopLimit) {
			////////console.warn("INSIDE THE LOOP");
          if (!this.isPaused) {
            const taskResult = await this.task(i);
           //console.log(taskResult);
            if(!taskResult){
				//console.error("UNDEFINED RESULT");
				          if (this.chainlink) {
            this.progressFunction(this.id, -1, this.label, this.stop.bind(this));
          }
				resolve(this.result);
				return;
			}
            //console.warn("PUSHING ", taskResult);
            this.result.push(taskResult);
          }

          if (iteratorCallback || (this.chainlink && i % skipIterations === 0)) {
            if (!this.isRunning) {
				//console.warn("SHUTTING DOWN WORKER");
              this.progressFunction(this.id, -1, this.label, this.stop.bind(this));
              resolve();
              return;
            }
//console.warn("UPDATING PROGRESS");
            this.progressFunction(this.id, i / (this.loopLimit - 1), this.label, this.stop.bind(this));
            //this.isPaused = this.progressFunction(this.id, "isPaused", this.label, this.stop.bind(this));
          }
		if(iteratorCallback){
			this.currentIteration = iteratorCallback(i);
		}else{
          this.currentIteration = i + 1;
        }
          setTimeout(() => loop(this.currentIteration), this.delay);
        } else {
			////////console.warn("OUTSIDE THE LOOP");
			//console.warn("FINISHED WITH ", this.result);
          if (this.chainlink) {
            this.progressFunction(this.id, -1, this.label, this.stop.bind(this));
          }
          resolve(this.result);
        }
      } catch (error) {
        if (this.chainlink) {
          this.progressFunction(this.id, -1, this.label, this.stop.bind(this));
        }
        reject(error);
      }
    };

    loop(this.currentIteration);
  });
}



  stop() {
    this.isRunning = false;
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
    this.run();
  }
async runThreaded(variables={}, classes=[]) {

  const results = [];
  const chunkSize = Math.ceil(this.loopLimit / this.numWorkers);
  const workers = [];
  
  const totalWorkers = Math.min(this.numWorkers, this.loopLimit);
  
  const progresses = [];
  for(let i = 0; i < totalWorkers; i++){
	  progresses[i] = 0;
  }
  for (let i = 0; i < totalWorkers; i++) {
	      	const start = i * chunkSize;
    	const end = Math.min(start + chunkSize, this.loopLimit);
    	//////console.log(`Worker ${i}: start = ${start}, end = ${end}`);
	   const workerPromise = this.runWorker(start, end, variables, classes, (progress) => {
		   progresses[i] = progress;
		   let totalProgress = 0;
		   		this.progressFunction(i+":"+this.id, progress, `Thread: ${i}`,  this.stop.bind(this));
		   		for(let j = 0; j < totalWorkers; j++){
					   totalProgress += progresses[j];
				   }
		   		this.progressFunction(this.id, totalProgress / totalWorkers, this.label, this.stop.bind(this));
		   });
workerPromise.then(resultArray => {
  results[i] = resultArray;
});
    workers.push(workerPromise);
  }

 await Promise.all(workers);
this.results = results.flat();
return this.results;
}

async runWorker(start, end, variables, classNames = [], progressCallback) {
	 const length = end-start;
  return new Promise((resolve) => {
    const variableNames = Object.keys(variables);
    const variableValues = Object.values(variables);
    const variableDeclarations = variableNames
      .map(
        (name, index) =>
          `let ${name} = ${JSON.stringify(variableValues[index])};`
      )
      .join("\n");

    const classStrings = {};
    classNames.forEach((className) => {
      const classString = eval(className).toString();
      classStrings[className] = classString;
    });

const taskString = `
    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    (async () => {
      const resultArray = [];
     
      const updateInterval = Math.max(Math.round(${length} * .01), 1);
      
      const delay = ${this.delay}; // Modify this value to change the delay

      for (let i = ${start}; i < ${end}; i++) {
        const taskResult = await (${this.task.toString()})(i);
      
        if(taskResult){
        
          resultArray.push(taskResult);
        }
        if(i % updateInterval === 0){
          self.postMessage({ status: 'progressUpdate', progress: (i - ${start})/(${length})  });
      
        }

        // Introduce delay between iterations
        await sleep(delay);
      }
      
      self.postMessage({ status: 'done', resultArray });
    })();
  `;

    const classImports = Object.keys(classStrings)
      .map((className) => {
        const classString = classStrings[className];
        return `const ${className} = ${classString};`;
      })
      .join("\n");

const workerBlob = new Blob(
  [
    classImports,
    `${variableDeclarations}
    self.onmessage = async (event) => {
      eval(${JSON.stringify(taskString)});
    };
  `,
  ],
  { type: "application/javascript" }
);

    const workerURL = URL.createObjectURL(workerBlob);
    const worker = new Worker(workerURL);

    const results = [];

    worker.postMessage({});

    worker.onmessage = (event) => {
      if (event.data.status === "partialResult") {
        results.push(event.data.taskResult);
      } else if (event.data.status === "progressUpdate") {
		 // //////console.warn("TRYING TO UPDATE PROGRESS");
        progressCallback(event.data.progress);
        } else if (event.data.status === "done") {
        resolve(event.data.resultArray);
        worker.terminate();
        progressCallback(-1);
        URL.revokeObjectURL(workerURL);
      }
  };
  });
}
}

