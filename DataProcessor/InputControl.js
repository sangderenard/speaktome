function showModal(content) {
  // Create the modal container
  const modal = document.createElement("div");
  modal.classList.add("modal");

  // Create the content container
  const contentContainer = document.createElement("div");
  contentContainer.classList.add("modal-content");
  contentContainer.appendChild(content);

  // Add the content container to the modal container
  modal.appendChild(contentContainer);

  // Add the modal container to the page
  document.body.appendChild(modal);

  return modal;
}

class DomainInputBox{
constructor(){
	this.id = Math.random();
this.initialBoxes = `<div>  <button type="button" id="add-input-pair">Add Input Pair</button>
  <button type="button" id="remove-input-pair">Remove Input Pair</button></div><div id="left-bottom-panel">
<div><div id="arrow" style="width: 0;
           height: 0px;
           border-left: 10px solid transparent;
           border-right: 10px solid transparent;
           border-bottom: 10px solid black;">
</div></div>
<div id="form-container">
<form>
 <!-- <button type="button" id="calculate-workout-plan">Calculate Workout Plan</button> -->
  <div id="input-container">
    <div class="inputPair" style="display: flex; justify-content: space-between; align-items: center;">
      <div style="width: 40%;">
        <label for="duration-1">Duration 1:</label>
        <input type="number" id="${this.id}-duration-1" value="1">
      </div>
      <div style="width: 40%;">
        <label for="power-1">Power 1:</label>
        <input type="number" id="${this.id}-power-1" value="1">
      </div>
    </div>
  </div>

</form>

</div>
<div><div id="arrowtwo" style="width: 0;
           height: 0px;
           border-left: 10px solid transparent;
           border-right: 10px solid transparent;
           border-top: 10px solid black;">
</div></div>
</div></div>`;
}
addActionListeners(){
	document.getElementById("duration-1").addEventListener("change", verifyInputs);
document.getElementById("power-1").addEventListener("change", verifyInputs);


document.querySelector('#add-input-pair').addEventListener('click', addInputPair);
document.querySelector('#remove-input-pair').addEventListener('click', removeInputPair);

let sortWorkoutArray = (w, s)=> {
switch (s) {
case "1": w.sort((a,b)=>a.duration-b.duration);break;
case "2": w.sort((a,b)=>b.duration-a.duration);break;
case "3": w.sort((a,b)=>a.power-b.power);break;
case "4": w.sort((a,b)=>b.power-a.power);break;
case "5": w.sort((a,b)=>a.duration!==b.duration?a.duration-b.duration:a.power-b.power);break;
case "6": w.sort((a,b)=>b.duration!==a.duration?b.duration-a.duration:b.power-a.power);break;
case "7": w.sort((a,b)=>a.power!==b.power?a.power-b.power:a.duration-b.duration);break;
case "8": w.sort((a,b)=>b.power!==a.power?b.power-a.power:b.duration-a.duration);break;
case "9": w.sort((a,b)=>a.duration!==b.duration?a.duration-b.duration:a.power-b.power);break;
case "10":w.sort((a,b)=>a.power!==b.power?a.power-b.power:a.duration-b.duration);break;
case "11":w.sort((a,b)=>b.power!==a.power?b.power-a.power:b.duration-a.duration);break;
case "12":w.sort(()=>Math.random()-0.5);break;
default: break;
}
return w;
}

}
verifyInputs = () => {
	  let durationInputs = [...document.querySelectorAll("input[id^='duration']")].map(i => Number(i.value));
	  let powerInputs = [...document.querySelectorAll("input[id^='power']")].map(i => Number(i.value));
	  const numInputs = durationInputs.length;
	  const validatedDurationInputs = validateValues(durationInputs, 'ascending');

	  let powerInputsValidated;
	  try {
	    powerInputsValidated = validateValues(powerInputs, 'descending');
	  } catch (error) {
	    return;
	  }

	    [...document.querySelectorAll("input[id^='duration']")].forEach((d, i) => d.value = validatedDurationInputs[i]);
	    [...document.querySelectorAll("input[id^='power']")].forEach((p, i) => p.value = powerInputsValidated[i]);

	    document.getElementById("graphPowerCurve").innerHTML = "";
	    copyStyles(document.getElementById("graph"), document.getElementById("graphPowerCurve"));
	    copyStyles(document.getElementById("graph"), document.getElementById("graphOne"));
	    copyStyles(document.getElementById("graph"), document.getElementById("graphFour"));
	    userSettings.powers = powerInputsValidated;
	    userSettings.durations = validatedDurationInputs;
	    let cumDuration = 0;
	    let newPoints = [];
	    let newIntervals = [];

	    for (let i = 0; i < userSettings.durations.length; i++) {
	    	  const point = new Point({ x: userSettings.durations[i], y: userSettings.powers[i] });
	    	  newPoints.push(point);
	    	  const interval = new Interval(cumDuration, userSettings.durations[i], point);
	    	  newIntervals.push(interval);
	    	  cumDuration += userSettings.durations[i];
	    	}

	    	primaryInput = new DataSet(newPoints, newIntervals);
	    createGraph(graphPowerCurve, [...Array(numInputs).keys()].map((i) => ({duration: validatedDurationInputs[i], power: powerInputsValidated[i]})), "duration");

	    const inputPairElements = document.querySelectorAll(".inputPair");
	    const firstInputPairElement = inputPairElements[0];
	    const offsetHeight = firstInputPairElement.offsetHeight;
	    document.getElementById("arrow").style.borderBottom = offsetHeight * inputPairCounter + "px solid";
	    document.getElementById("arrowtwo").style.borderTop = offsetHeight * inputPairCounter + "px solid";
	    saveSettings();
	    calculateWorkoutPlan(primaryInput);

	};

	
	removeInputPair = () => {
	if (inputPairCounter > 1) {
	const inputContainer = document.querySelector('#input-container');
	inputContainer.removeChild(inputContainer.lastChild);
	inputPairCounter--;
	this.verifyInputs();
	}

	};
	}
	
	
	
	
	
	class InputControl {
	  constructor() {
	    this.elements = [];
	    this.inputBoxes = [];
	    this.processId;
	   this.liveMode = false;
  this.liveModeInterval = null;
  this.clickTimeout;

	  }
static Debounce(func, wait= 300) {
  let timeout;

  return function (...args) {
    const context = this;
    clearTimeout(timeout);

    timeout = setTimeout(() => {
      func.apply(context, args);
    }, wait);
  };
}
async createInputs(functionChainLink, inputParams, inputTypes, labels, config ={
    includeHistoryButtons: true,
    includePreviewGraph: true,
    includeProgressIndicators: true,
    includeOutputMapping: true,
    outputMappingInput: standardAxes,
    outputMappingOutput: standardAxes,
  }) {
  const {
    includeHistoryButtons = true,
    includePreviewGraph = true,
    includeProgressIndicators = true,
    includeOutputMapping = true,
    outputMappingInput = standardAxes,
    outputMappingOutput = standardAxes,
  } = config;
  if(functionChainLink){
  functionChainLink.inputControl = this;
  this.functionChainLink = functionChainLink;
}
  const inputElements = [];
  const container = document.createElement("div");
  container.style.display = "flex";

  // Create function parameter inputs on the left
  const inputContainer = document.createElement("div");
  inputContainer.style.display = "flex";
  inputContainer.style.flexDirection = "column";
  inputContainer.style.width = "auto";
  container.appendChild(inputContainer);
		  for (let i = 0; i < inputTypes.length; i++) {
		    const type = inputTypes[i];
		    const label = labels[i];
		    const defaultValue = inputParams[i];

		    const inputBox = document.createElement("div");
		    inputBox.classList.add("input-box");

		    switch (type) {
		      case "text":
		        inputElements.push(this.createTextInput(inputBox, label, defaultValue));
		        break;
		      case "number":
		        inputElements.push(this.createNumberInput(inputBox, label, defaultValue));
		        break;
		      case "range":
		        inputElements.push(
		          this.createRangeInput(inputBox, label, defaultValue[0], defaultValue[1], defaultValue[2], defaultValue[3])
		        );
		        break;
		      case "radio":
		        const [options, defaultOption] = defaultValue;
		        inputElements.push(
		          ...this.createRadioInputGroup(inputBox, label, options, defaultOption)
		        );
		        break;
		      case "checkbox":
		        inputElements.push(this.createCheckboxInput(inputBox, label, defaultValue));
		        break;
		      case "select":
		        inputElements.push(this.createSelectInput(inputBox, label, defaultValue[0], defaultValue[1]));
		        break;
		      case "axisSelect":
		        inputElements.push(this.createAxisSelectInput(inputBox, label, defaultValue));
		        break;
		        case "axesSelect":
		        inputElements.push(this.createAxesSelectInput(inputBox, label, defaultValue));
		        break;
		      case "custom":
		        inputElements.push(this.createCustomControl(inputBox, label, defaultValue));
		        break;
		      case "checkboxgroup":
		        inputElements.push(...this.createCheckboxGroupInput(inputBox, label, defaultValue[0], defaultValue[1]));
		        break;
		      case "file":
		        inputElements.push(this.createFileInput(inputBox, label));
		        break;
		        case "fileSelector":
					inputElements.push(await this.createFileSelector(inputBox, label, defaultValue));
					break;
		      case "pointSelect":
		    	  inputElements.push(this.createPointInputElement(inputBox, label, defaultValue));
		    	  break;
		      case "pointsSelect":
		    	  inputElements.push(this.createPointsInputElement(inputBox, label, defaultValue));
		    	  break;
		      default:
		        throw new Error(`Invalid input type: ${type}`);
		    }
		    if(functionChainLink){
		    	this.inputBoxes.push(inputBox);
		    }
		    inputContainer.appendChild(inputBox);
		  }
	if(functionChainLink){
  		this.elements = { inputElements };
	}
  if (includeHistoryButtons) {
    this.historyButtons = this.createHistoryButtons(inputElements, functionChainLink);
    inputContainer.appendChild(this.historyButtons);
  }

	if(includeOutputMapping){
		let mapLabel = 'output';
		if(!includeHistoryButtons){
			mapLabel = 'specialMapping';
		}
  		const mappingBox = document.createElement("div");
  		const mappingBoxContent = document.createElement("div");
  		const outputMapping = this.createPopupDiv("Output Mapping", mappingBox, this.createCheckboxGrid(mappingBoxContent, config.outputMappingInput, config.outputMappingOutput, mapLabel));
  		inputContainer.appendChild(mappingBox);
	}

  if (includeProgressIndicators) {
    this.progressBarManager = new ProgressBarManager();
    const progressContainer = this.progressBarManager.createProgressBar(this);
    container.appendChild(progressContainer);
    	functionChainLink.processID = this.processID;
		functionChainLink.progress = this.progress;
}
  if (includePreviewGraph) {
    const graphContainer = this.createPreviewGraphOutput();
    graphContainer.classList.add("input-box");
    container.appendChild(graphContainer);
  }

  return {
    container,
    inputContainer,
    inputBoxes: this.inputBoxes,
    inputElements: inputElements,
    graphFunc: includePreviewGraph?this.preview.bind(this):null
  };
}


 interruptingPrompt(inputParams, inputTypes, labels, timeout, config, callback = null) {
	   const {
    includeHistoryButtons = false,
    includePreviewGraph = false,
    includeProgressIndicators = false,
    includeOutputMapping = false,
    outputMappingInput = standardAxes,
    outputMappingOutput = standardAxes,
  } = config
  return new Promise(async (resolve) => {
    const promptContainer = document.createElement("div");
    promptContainer.style.display = "flex";
    promptContainer.style.flexDirection = "column";
    promptContainer.style.width = "auto";
console.error("before createInputs");

    const inputs = await this.createInputs(null, inputParams, inputTypes, labels, {includeHistoryButtons, includePreviewGraph, includeProgressIndicators, includeOutputMapping, outputMappingInput, outputMappingOutput});
console.error("after createInputs");
let okButton, cancelButton, buttonContainer;
try{
    promptContainer.appendChild(inputs.container);

    buttonContainer = document.createElement("div");
    buttonContainer.style.display = "flex";
    buttonContainer.style.justifyContent = "space-between";

    okButton = document.createElement("button");
    okButton.innerText = "ok";
    buttonContainer.appendChild(okButton);

    cancelButton = document.createElement("button");
    cancelButton.innerText = "cancel";
    buttonContainer.appendChild(cancelButton);

    promptContainer.appendChild(buttonContainer);
    }catch(error){
		console.error(error);
	}
console.error("before modal");
    const modal = showModal(promptContainer);
console.error("after modal");
    let timeoutId = null;
let timeoutCounter = timeout;

const startTimeout = () => {
  timeoutId = setTimeout(() => {
    clearTimeout(timeoutId);
    timeoutId = null;
    okButton.innerText = "ok";
    handleOk();
  }, 1000);

  updateOkButton();
};

const updateOkButton = () => {
  if (timeoutId) {
    okButton.innerText = `ok (${timeoutCounter--})`;
    setTimeout(updateOkButton, 1000);
  }
};

const handleOk = () => {
  const values = inputs.inputElements.map((input) => input.value);
  const result = callback ? callback(values) : values;
  resolve(result);
  modal.remove();
};

const handleCancel = () => {
  resolve(null);
  modal.remove();
};

okButton.addEventListener("click", () => {
  if (!timeoutId) {
    handleOk();
  }
});

cancelButton.addEventListener("click", handleCancel);

promptContainer.addEventListener("mousemove", () => {
  if (timeoutId) {
    clearTimeout(timeoutId);
    timeoutId = null;
    okButton.innerText = "ok";
  }
});

startTimeout();
});
}
		createProgressBar(inputControl){
			
			inputControl.progress = (process, progress) => {
				
			}
		}
		
		createHistoryButtons(inputElements, functionChainLink) {
			this.commitHistory = [];
			this.redoHistory = [];
			////console.error("PASSING THESE INPUTS ", inputElements);
			this.currentValues = FunctionChain.getUserInputStatic(inputElements);
			////console.log(this.currentValues);
			
			 
			
			
  const commitValues = (inputElements, functionChainLink) => {
    // Store previous values
    this.previousValues = this.currentValues.slice();
   // this.currentValues = inputElements.map((input) => input.value);
this.currentValues = FunctionChain.getUserInputStatic(inputElements);
    // Compare previous values to current values and ignore if they are the same
    if (this.livemode && this.previousValues.toString() === this.currentValues.toString()) {
		////console.warn("VALUES ARE THE SAME");
      return;
    }

    // Add the new values to the commit history
    this.commitHistory.push(this.currentValues);

    // Verify inputs
    functionChainLink.verifyInputs();
  };

  const revertValues = (inputElements) => {
    // Revert to the last committed values or default values
    const lastCommit = this.commitHistory.length > 0 ? this.commitHistory[this.commitHistory.length - 1] : this.defaultValues;

    inputElements.forEach((input, index) => {
      input.value = lastCommit[index];
    });
  };

  const undoValues = (inputElements) => {
    if (this.commitHistory.length <= 1) {
      return; // Cannot undo if there is no history or only one commit
    }

    // Remove the last commit from the history
    const lastCommit = this.commitHistory.pop();

    // Add it to the redo history
    this.redoHistory.push(lastCommit);

    // Get the new last commit from the history and apply it
    const previousCommit = this.commitHistory[this.commitHistory.length - 1];
    inputElements.forEach((input, index) => {
      input.value = previousCommit[index];
    });
    commitValues();
  };

  const redoValues = (inputElements) => {
    if (this.redoHistory.length === 0) {
      return; // Cannot redo if there is no redo history
    }

    // Get the last redo history entry and apply it
    const redoCommit = this.redoHistory.pop();
    inputElements.forEach((input, index) => {
      input.value = redoCommit[index];
    });

    // Add the redo history entry back to the commit history
    this.commitHistory.push(redoCommit);
  }; 
  const buttonsContainer = document.createElement("div");
  buttonsContainer.style.display = "flex";
  buttonsContainer.style.justifyContent = "space-between";
  buttonsContainer.style.marginTop = "10px";



  const toggleLiveMode = () => {
    this.liveMode = !this.liveMode;

    if (this.liveMode) {
      commitButton.style.backgroundColor = "red";
      commitButton.style.animation = "blink 1s linear infinite";
      this.liveModeInterval = setInterval(() => {
        commitValues(inputElements, functionChainLink);
      }, 500); // Change 500 to the desired interval in milliseconds
    } else {
      commitButton.style.backgroundColor = "";
      commitButton.style.animation = "";
      clearInterval(this.liveModeInterval);
      this.liveModeInterval = null;
    }
  };

  const singleClickHandler = () => {
    if (this.liveMode) {
      toggleLiveMode();
    } else {
      // Call commit method
      commitValues(inputElements, functionChainLink);
    }
  };

  const doubleClickHandler = () => {
    if (!this.liveMode) {
      toggleLiveMode();
    }
  };

  const commitButton = document.createElement("button");
  commitButton.innerText = "Commit";
  commitButton.addEventListener("click", (event) => {
    event.preventDefault();
    clearTimeout(this.clickTimeout);
    this.clickTimeout = setTimeout(() => {
      singleClickHandler();
    }, 300);
  });
  commitButton.addEventListener("dblclick", (event) => {
    event.preventDefault();
    clearTimeout(this.clickTimeout);
    doubleClickHandler();
  });
  buttonsContainer.appendChild(commitButton);
  const revertButton = document.createElement("button");
  revertButton.innerText = "Revert";
  revertButton.onclick = () => {
    // Call revert method
    revertValues(inputElements);
  };
  buttonsContainer.appendChild(revertButton);

  const undoButton = document.createElement("button");
  undoButton.innerText = "Undo";
  undoButton.onclick = () => {
    // Call undo method
    undoValues(inputElements);
  };
  buttonsContainer.appendChild(undoButton);

  const redoButton = document.createElement("button");
  redoButton.innerText = "Redo";
  redoButton.onclick = () => {
    // Call redo method
    redoValues(inputElements);
  };
  buttonsContainer.appendChild(redoButton);

  return buttonsContainer;
}
	  createTextInput(inputBox, label, defaultValue) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const inputElement = document.createElement("input");
	    inputElement.type = "text";
	    inputElement.value = defaultValue;

	    inputBox.appendChild(labelElement);
	    inputBox.appendChild(inputElement);

	    return inputElement;
	  }

	  createNumberInput(inputBox, label, defaultValue) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const inputElement = document.createElement("input");
	    inputElement.type = "number";
	    inputElement.value = defaultValue;

	    inputBox.appendChild(labelElement);
	    inputBox.appendChild(inputElement);

	    return inputElement;
	  }

createRangeInput(inputBox, label, min, max, step, defaultValue) {
  const labelElement = document.createElement("label");
  labelElement.textContent = label;

  const inputElement = document.createElement("input");
  inputElement.type = "range";
  inputElement.value = defaultValue;
  inputElement.min = min;
  inputElement.max = max;
  inputElement.step = step;

  const rangeDisplayElement = document.createElement("input");
  rangeDisplayElement.type = "text";
  rangeDisplayElement.value = defaultValue;
  rangeDisplayElement.style.width = "4em";
  rangeDisplayElement.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    if (value >= min && value <= max) {
      inputElement.value = value;
      inputElement.dispatchEvent(new Event("input"));
    }
  });

  const minDisplayElement = document.createElement("input");
  minDisplayElement.type = "text";
  minDisplayElement.value = min;
  minDisplayElement.style.width = "3em";
  minDisplayElement.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    if (value <= max) {
      inputElement.min = value;
      min = value;
    } else {
      event.target.value = min;
    }
  });

  const maxDisplayElement = document.createElement("input");
  maxDisplayElement.type = "text";
  maxDisplayElement.value = max;
  maxDisplayElement.style.width = "3em";
  maxDisplayElement.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    if (value >= min) {
      inputElement.max = value;
      max = value;
    } else {
      event.target.value = max;
    }
  });

  inputElement.addEventListener("input", (event) => {
    rangeDisplayElement.value = event.target.value;
  });

  inputBox.appendChild(labelElement);
  inputBox.appendChild(minDisplayElement);
  inputBox.appendChild(inputElement);
  inputBox.appendChild(maxDisplayElement);
  inputBox.appendChild(rangeDisplayElement);

  return inputElement;
}

	  createRadioInputGroup(inputBox, label, options, defaultOption) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const inputElements = [];
const uniqueId = Math.random();
	    options.forEach((option) => {
	      const inputElement = document.createElement("input");
	      inputElement.type = "radio";
	      inputElement.name = label+uniqueId;
	      inputElement.value = option;
	      inputElement.checked = option === defaultOption;

	      const optionLabelElement = document.createElement("label");
	      optionLabelElement.textContent = option;

	      inputBox.appendChild(inputElement);
	      inputBox.appendChild(optionLabelElement);

	      inputElements.push(inputElement);
	    //  inputElements.push(optionLabelElement);
	    });

	    inputBox.appendChild(labelElement);

	    return inputElements;
	  }

	  createCheckboxInput(inputBox, label, defaultValue) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const uniqueNewYork = Math.random();
	    const inputElement = document.createElement("input");
	    inputElement.type = "checkbox";
	    inputElement.name = uniqueNewYork;
	    inputElement.checked = defaultValue;

	    inputBox.appendChild(inputElement);
	    inputBox.appendChild(labelElement);

	    return inputElement;
	  }

	  createSelectInput(inputBox, label, options, defaultValue) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const selectElement = document.createElement("select");
	    selectElement.value = defaultValue;

	    options.forEach((option) => {
	      const optionElement = document.createElement("option");
	      optionElement.value = option;
	      optionElement.text = option;

	      if (option === defaultValue) { // add this line to select the default value
	            optionElement.selected = true;
	        }
	      
	      selectElement.appendChild(optionElement);
	    });

	    inputBox.appendChild(labelElement);
	    inputBox.appendChild(selectElement);

	    return selectElement;
	  }
	  
	  createAxisSelectInput(inputBox, label, defaultValue) {
		  const selectOptions = [''].concat(standardAxes);
		  const selectElement = this.createSelectInput(inputBox, label, [...selectOptions, 'none'], defaultValue);
		  return selectElement;
		}
			  createAxesSelectInput(inputBox, label, defaultValues) {
		  		  const container = document.createElement("div");
		  let inputElements = [];
		  for(let i = 0; i < defaultValues.length; i++){
			  inputElements.push(this.createAxisSelectInput(container, `Axis ${i}`, defaultValues[i]));
			  
		  }

		  const condensedInput = document.createElement("input");
		  const condenserFunc = (event) => {
			    const values = inputElements.map(inputElement => inputElement.value);
		
			  	  let output = "";
				  for(let i = 0; i < values.length; i++){
					  output += values[i]+'|';
				  }
			    	
			    condensedInput.value = output;
	
			    //condensedInput.dispatchEvent(new Event('input'));
		  }
		  
  
		  
		  for(let i = 0; i < inputElements.length; i++){
			  container.appendChild(inputElements[i]);
			  inputElements[i].addEventListener('input', condenserFunc);
		}
		  inputElements[0].dispatchEvent(new Event('input'));
		  inputBox.appendChild(this.boxAControlSet(container, "Axes"));
		  
		  
		  return condensedInput;
		}
	  createPointsInputElement(inputBox, label, defaultValues){
		  const container = document.createElement("div");
		  let inputElements = [];
		  for(let i = 0; i < defaultValues[0]; i++){
			  inputElements.push(this.createPointInputElement(container, `Point ${i}`, defaultValues[1]));
			  
		  }

		  const condensedInput = document.createElement("input");
		  const condenserFunc = (event) => {
			    const values = inputElements.map(inputElement => inputElement.value);
		
			  	  let output = "";
				  for(let i = 0; i < values.length; i++){
					  output += values[i]+'|';
				  }
			    	
			    condensedInput.value = output;
	
			    condensedInput.dispatchEvent(new Event('input'));
		  }
		  
    const initialValue = "";
  condensedInput.value = initialValue;
		  
		  for(let i = 0; i < inputElements.length; i++){
			  container.appendChild(inputElements[i]);
			  inputElements[i].addEventListener('input', condenserFunc);
		}
		  
		  inputBox.appendChild(this.boxAControlSet(container, "Points"));
		  
		  
		  return condensedInput;
	  }
	  createPointInputElement(inputBox, label, defaultValue) {
		  const container = document.createElement("div");
		  
		  let inputElements = [];
		  for(let i = 0; i < defaultValue; i++ ){
			  
			 inputElements.push(this.createAxisSelectInput(container, `axis ${i}`, 'none'));
			  inputElements.push(this.createNumberInput(container, "=", 0));
	  		}

		  
		  const condensedInput = document.createElement("input");
		  const condenserFunc = (event) => {
			    const values = inputElements.map(inputElement => inputElement.value);
		
			  	  let output = "";
				  for(let i = 0; i < values.length; i+=2){
					  output += values[i]+':'+values[i+1]+',';
				  }
			    	
			    condensedInput.value = output;
	
			    condensedInput.dispatchEvent(new Event('input'));
		  }

		  for(let i = 0; i < inputElements.length; i++){
			  container.appendChild(inputElements[i]);
			  inputElements[i].addEventListener('input', condenserFunc);
	}
		  inputBox.appendChild(this.boxAControlSet(container, "Point"));
		  
		  return condensedInput;
		  
		}

	  createCustomControl(inputBox, label, defaultValue) {
	    const labelElement = document.createElement("label");
	    labelElement.textContent = label;

	    const controlElement = document.createElement("div");
	    controlElement.innerHTML = defaultValue;

	    inputBox.appendChild(labelElement);
	    inputBox.appendChild(controlElement);

	    return controlElement;
	    
	  }
	  createFileInput(inputBox, label) {
		    const labelElement = document.createElement("label");
		    labelElement.textContent = label;

		    const inputElement = document.createElement("input");
		    inputElement.type = "file";

		    inputBox.appendChild(labelElement);
		    inputBox.appendChild(inputElement);

		    return inputElement;
		}

static async getFileObject(filePath) {
  return fetch(filePath)
    .then(response => response.blob())
    .then(blob => {
      const file = new File([blob], filePath.split('/').pop(), { type: blob.type });
      return file;
    });
}
		async createFileSelector(inputBox, label, folderLocation) {
  // Create the label for the file selector
  //console.error(folderLocation);
  const labelElement = document.createElement("label");
  labelElement.textContent = label;

  // Create the file input using the createFileInput function
  const fileInputElement = this.createFileInput(inputBox, "Select a file");

  // Create the password input
  const passwordInputElement = document.createElement("input");
  passwordInputElement.type = "password";
  passwordInputElement.placeholder = "Enter password";

  // Create the ID input
  const idInputElement = document.createElement("input");
  idInputElement.type = "text";
  idInputElement.placeholder = "Enter ID";
  
  // Create the folder input
  const folderInputElement = document.createElement("input");
  folderInputElement.type = "text";
  folderInputElement.placeholder = "Enter folder";

const serverResponse = document.createElement("textarea");

	//Create the listing button
	const listingButtonElement = document.createElement("button");
	listingButtonElement.textContent = "Get Listing";
	listingButtonElement.onclick = async () => {
			const password = passwordInputElement.value;
	const username = idInputElement.value;
	const folder = folderInputElement.value;
	let response;
					try{
				response = await ServerCommunication.buildDirectorySelector(username, password, folder, selectElement, serverResponse);
			}catch(error){
				console.log("error rebuilding directory", error);
			}
	};

  // Create the upload button
  const uploadButtonElement = document.createElement("button");
  uploadButtonElement.textContent = "Upload";
  uploadButtonElement.onclick = async () => {
  try{
    // Get the selected file from the file input
    const file = fileInputElement.files[0];
// Get the password and ID values from the input elements
	const password = passwordInputElement.value;
	const username = idInputElement.value;
	const folder = folderInputElement.value;
	let uploadSuccess, response; 
	
		try{
			uploadSuccess = await ServerCommunication.uploadFile(file, password, username, folder, serverResponse);
		}catch(error){
			console.error("error uploading", error);
		}
    // If upload succeeded, rebuild the file selector with updated files
   		if (uploadSuccess) {
			try{
				response = await ServerCommunication.buildDirectorySelector(username, password, folder, selectElement, serverResponse);
			}catch(error){
				console.log("error rebuilding directory", error);
			}
		}
    }catch(error){
		console.log("general error in upload process: ", error);
  	}
  };

  // Create the select element for the file list
  const selectElement = document.createElement("select");
selectElement.setAttribute("size", "8");

 //await ServerCommunication.buildDirectorySelector(folderLocation, selectElement);

  // Add all the input elements to the input box
  inputBox.appendChild(labelElement);
  if(selectElement.options){
	  inputBox.appendChild(selectElement);
	  }
  inputBox.appendChild(passwordInputElement);
  inputBox.appendChild(idInputElement);
  inputBox.appendChild(folderInputElement);
  inputBox.appendChild(listingButtonElement);
  inputBox.appendChild(uploadButtonElement);
  inputBox.appendChild(serverResponse);
console.warn("finished creating file selector");
  return selectElement;
}
	  createCheckboxGroupInput(inputBox, label, options, defaultOptions) {
		    const labelElement = document.createElement("label");
		    labelElement.textContent = label;

		    const inputElements = [];
		    const uniqueId = Math.random();

		    for (const option of options) {
		        const inputElement = document.createElement("input");
		        inputElement.type = "checkbox";
		        inputElement.id = `${label}${option}${uniqueId}`;
		        inputElement.name = label;
		        inputElement.value = option;
		        inputElements.push(inputElement);

		        const optionLabelElement = document.createElement("label");
		        optionLabelElement.textContent = option;
		        optionLabelElement.setAttribute("for", `${label}${option}${uniqueId}`);
		       // inputElements.push(optionLabelElement);

		        inputBox.appendChild(inputElement);
		        inputBox.appendChild(optionLabelElement);
		    }

		    inputBox.appendChild(labelElement);

		    return inputElements;
		}

	  boxAControlSet(controlContainer, headerText){
		  const div = document.createElement('div');

		// create the header element
		const header = document.createElement('div');
		header.textContent = headerText;

		// create the content element
		const content = controlContainer;
		
		// add the header and content to the div
		div.appendChild(header);
		div.appendChild(content);

		// set initial state of content to hidden
		content.style.display = 'none';

		// add click event listener to the header
		header.addEventListener('click', () => {
		  if (content.style.display === 'none') {
		    content.style.display = 'block';
		  } else {
		    content.style.display = 'none';
		  }
		});
		return div;
	  }
	  createCollapsibleDiv(container, content) {
  const wrapper = document.createElement('div');
  wrapper.style.position = 'relative';
  wrapper.style.height = '100%';
  wrapper.style.width = '100%';

  const smallBar = document.createElement('div');
  smallBar.style.position = 'absolute';
  smallBar.style.bottom = '0';
  smallBar.style.width = '100%';
  smallBar.style.height = '20px';
  smallBar.style.backgroundColor = 'gray';
  smallBar.style.cursor = 'pointer';
  smallBar.innerHTML = 'Click to expand';


  content.style.display = 'none';
  content.style.position = 'absolute';
  content.style.bottom = '20px';
  content.style.width = '100%';
  content.style.height = '200px';
  content.style.backgroundColor = 'lightgray';


  smallBar.addEventListener('click', () => {
    if (content.style.display === 'none') {
      content.style.display = 'block';
      smallBar.innerHTML = 'Click to collapse';
    } else {
      content.style.display = 'none';
      smallBar.innerHTML = 'Click to expand';
    }
  });

  wrapper.appendChild(smallBar);
  wrapper.appendChild(content);
  container.appendChild(wrapper);
}
createPopupDiv(label, container, content) {
  const popupWrapper = document.createElement('div');
  popupWrapper.style.position = 'fixed';
  popupWrapper.style.top = '0';
  popupWrapper.style.left = '0';
  popupWrapper.style.width = '100%';
  popupWrapper.style.height = '100%';
  popupWrapper.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  popupWrapper.style.display = 'none';
  popupWrapper.style.zIndex = '9999';

  const popup = document.createElement('div');
  popup.style.position = 'absolute';
  popup.style.top = '50%';
  popup.style.left = '50%';
  popup.style.transform = 'translate(-50%, -50%)';
  popup.style.backgroundColor = 'white';
  popup.style.padding = '20px';
  popup.style.borderRadius = '5px';

  const closeButton = document.createElement('button');
  closeButton.style.position = 'absolute';
  closeButton.style.top = '10px';
  closeButton.style.right = '10px';
  closeButton.style.border = 'none';
  closeButton.style.background = 'none';
  closeButton.style.fontSize = '24px';
  closeButton.style.fontWeight = 'bold';
  closeButton.style.cursor = 'pointer';
  closeButton.innerHTML = '&times;';

  closeButton.addEventListener('click', () => {
    popupWrapper.style.display = 'none';
  });

  const triggerButton = document.createElement('button');
  triggerButton.textContent = label;
  triggerButton.style.cursor = 'pointer';

  triggerButton.addEventListener('click', () => {
    popupWrapper.style.display = 'block';
  });

  popup.appendChild(closeButton);
  popup.appendChild(content);
  popupWrapper.appendChild(popup);
  container.appendChild(triggerButton);
  container.appendChild(popupWrapper);
}

	 createCheckboxGrid(container, columns, rows, mapID) {
    const updateMappingData = () => {
        let mapping = '';

        for (let r = 1; r < table.rows.length; r++) {
            for (let c = 1; c < table.rows[r].cells.length; c++) {
                if (table.rows[r].cells[c].firstChild.checked) {
                    mapping += `${rows[r - 1]}:${columns[c - 1]},`;
                }
            }
        }

        mappingData.value = mapping.slice(0, -1);
        updateChainlink(mappingData.value, mapID);
    }

	const updateChainlink=(mapping , mapID)=>{
		//console.error(mapping);
		if(!this.functionChainLink?.mappings){
			this.functionChainLink.mappings = {};
		}
		this.functionChainLink.mappings[mapID] = mapping;
	}

    const table = document.createElement('table');
    const mappingData = document.createElement('input');
    mappingData.type = 'text';
    mappingData.style.display = 'none';


    // Create header row
    const headerRow = table.insertRow();
    headerRow.insertCell(); // Empty cell for the top-left corner

    for (const column of columns) {
        const headerCell = headerRow.insertCell();
        headerCell.textContent = column;
    }

    // Create rows with checkboxes
    for (const row of rows) {
        const tableRow = table.insertRow();
        const rowHeader = tableRow.insertCell();
        rowHeader.textContent = row;

        for (let c = 0; c < columns.length; c++) {
            const cell = tableRow.insertCell();
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
                  // Check the checkbox if the row and column match
      
            checkbox.addEventListener('change', updateMappingData);
            cell.appendChild(checkbox);
            if (row === columns[c]) {
        checkbox.checked = true;
        updateMappingData();
      }
        }
    }

    container.appendChild(table);
    container.appendChild(mappingData);
    return container;
}
  
	  createPreviewGraphOutput(width = 200, height = 200){
		  const graphContainer = document.createElement("div");

		  // Create controls for graph
		  const controls = document.createElement("div");
		  controls.style.display = "flex";
		  controls.style.flexDirection = "column";
		  controls.style.width = "200px";

		  // Select box for polar or rectangular
		  const coordSystemSelect = this.createSelectInput(controls, "Coordinate System", ["Rectangular", "Polar"], "Rectangular");
		const drawWidthRange = this.createRangeInput(controls, "Draw Width", 1, 20, 1, 1);
		  const graphTypeSelect = this.createSelectInput(controls, "Graph Type", ["bar", "scatter", "line", "interval"], "line");
		    const scaleModeSelect = this.createSelectInput(controls, "Scale Mode", ["normal", "fit", "zoom"], "normal");
		    const scaleZoomRange = this.createRangeInput(controls, "Zoom", .01,100,.01,1);
		  // Select box for domain axis
		  const domainAxisSelect = this.createAxisSelectInput(controls, "Domain Axis", "x");

		  // Select boxes for range axes
		  const rangeAxisSelects = [];
		  const defaultRanges = ['y','none', 'none', 'none', 'none', 'none'];
		  for (let i = 0; i < 6; i++) {
		    const axisSelect = this.createAxisSelectInput(controls, `Range Axis ${i+1}`, defaultRanges[i]);
		    rangeAxisSelects.push(axisSelect);
		  }

		  // Select box for domain resolution
		  const resolutionSelect = this.createSelectInput(controls, "Resolution", [1, 2, 5, 10, 20, 50, 100], 1);

		  graphContainer.appendChild(this.boxAControlSet(controls, "Graph Controls"));

		  // Create canvas for graph
		  const canvas = document.createElement("canvas");
		  canvas.width = width;
		  canvas.height = height;
		  canvas.style.border = "1px solid black";
			const    dataTextArea = document.createElement('textarea');
			    dataTextArea.style.width = '100%';
			    dataTextArea.style.height = '200px';
			    
		  graphContainer.appendChild(canvas);

		  // Define graph object and preview function
		  const graph = new Graph(canvas, dataTextArea);
		  
			    
			    graphContainer.appendChild(dataTextArea);
		  this.preview = (dataset) => {
			if(!dataset){
				 return;
			  }
			  this.lastDataset = dataset;
			  
			  const resolution = parseInt(resolutionSelect.value);
			  
			  const reducedPoints = dataset.points.filter((p, i) => i % resolution === 0);
			  
			  const graphDataset = new DataSet(reducedPoints, this.lastDataset.intervals || null);
			  
			  graph.drawGraph(graphDataset||reducedPoints, {
				  width: drawWidthRange.value,
				  scaleMode: scaleModeSelect.value,
				  scaleZoom: scaleZoomRange.value,
				  progress: this.progress,
			    graphType: graphTypeSelect.value,
			    coordSystem: coordSystemSelect.value,
			    domainAxis: domainAxisSelect.value,
			    rangeAxes: rangeAxisSelects.map(s => s.value)
			  });
			};

		  // Add event listeners to controls to trigger preview
		  const controlElements = [drawWidthRange, scaleModeSelect, scaleZoomRange, coordSystemSelect, graphTypeSelect, domainAxisSelect, ...rangeAxisSelects, resolutionSelect];
		  controlElements.forEach(element => {
		    element.addEventListener('change', () => {
		      if (this.lastDataset) {
		        this.preview(this.lastDataset);
		      }
		    });
		  });

		  return graphContainer;
		}
}