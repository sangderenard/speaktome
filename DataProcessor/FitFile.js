const fitMetadataFields = [
  'developerDataIndex',
  'manufacturer',
  'product',
  'serialNumber',
  'event',
  'eventType',
  'fieldDefinitionNumber',
  'fitBaseTypeId',
  'fieldName',
  'nativeMesgNum',
  'messageIndex',
  'startTime',
  'totalElapsedTime',
  'type',
  'timeCreated',
  'deviceIndex',
  'productName',
  'softwareVersion',
  'applicationId',
  'applicationVersion',
  'nativeFieldNum',
  'sport',
  'subSport',
  'firstLapIndex',
  'numLaps',
  'numSessions',
  'localTimestamp',
  'totalTimerTime'
];
const BaseType = {
    ENUM: 0x00,
    SINT8: 0x01,
    UINT8: 0x02,
    SINT16: 0x83,
    UINT16: 0x84,
    SINT32: 0x85,
    UINT32: 0x86,
    STRING: 0x07,
    FLOAT32: 0x88,
    FLOAT64: 0x89,
    UINT8Z: 0x0A,
    UINT16Z: 0x8B,
    UINT32Z: 0x8C,
    BYTE: 0x0D,
    SINT64: 0x8E,
    UINT64: 0x8F,
    UINT64Z: 0x90
};
const BaseTypeDefinitions = {
    0x00: { size: 1, type: BaseType.ENUM, invalid: 0xFF },
    0x01: { size: 1, type: BaseType.SINT8, invalid: 0x7F },
    0x02: { size: 1, type: BaseType.UINT8, invalid: 0xFF },
    0x83: { size: 2, type: BaseType.SINT16, invalid: 0x7FFF },
    0x84: { size: 2, type: BaseType.UINT16, invalid: 0xFFFF },
    0x85: { size: 4, type: BaseType.SINT32, invalid: 0x7FFFFFFF },
    0x86: { size: 4, type: BaseType.UINT32, invalid: 0xFFFFFFFF },
    0x07: { size: 1, type: BaseType.STRING, invalid: 0x00 },
    0x88: { size: 4, type: BaseType.FLOAT32, invalid: 0xFFFFFFFF },
    0x89: { size: 8, type: BaseType.FLOAT64, invalid: 0xFFFFFFFFFFFFFFFF },
    0x0A: { size: 1, type: BaseType.UINT8Z, invalid: 0x00 },
    0x8B: { size: 2, type: BaseType.UINT16Z, invalid: 0x0000 },
    0x8C: { size: 4, type: BaseType.UINT32Z, invalid: 0x00000000 },
    0x0D: { size: 1, type: BaseType.BYTE, invalid: 0xFF },
    0x8E: { size: 8, type: BaseType.SINT64, invalid: 0x7FFFFFFFFFFFFFFF },
    0x8F: { size: 8, type: BaseType.UINT64, invalid: 0xFFFFFFFFFFFFFFFF },
    0x90: { size: 8, type: BaseType.UINT64Z, invalid: 0x0000000000000000 },
};

const FieldTypeToBaseType = {
    "sint8": BaseType.SINT8,
    "uint8": BaseType.UINT8,
    "sint16": BaseType.SINT16,
    "uint16": BaseType.UINT16,
    "sint32": BaseType.SINT32,
    "uint32": BaseType.UINT32,
    "string": BaseType.STRING,
    "float32": BaseType.FLOAT32,
    "float64": BaseType.FLOAT64,
    "uint8z": BaseType.UINT8Z,
    "uint16z": BaseType.UINT16Z,
    "uint32z": BaseType.UINT32Z,
    "byte": BaseType.BYTE,
    "sint64": BaseType.SINT64,
    "uint64": BaseType.UINT64,
    "uint64z": BaseType.UINT64Z
};
class FitFile{
constructor(file, chainlink){
		if(chainlink){
		this.processID = chainlink.inputControl.progressBarManager.getID();
		this.chainlink = chainlink;
	}
	//this.progressFunc = progressFunc;
	////////////console.error(this.progressFunc);
	this.localMessageDefinitions = {};
	this.developerDataDefinitions = {};
	this.fieldsWithSubFields = [];
	this.fieldsToExpand = [];
//	this.processID = Math.random();

this.developers = {};

}

computeDataSize(fields, definition, arch) {
  let dataSize = 0;
  //////////////console.error("COMPUTING DATA SIZE: ",definition);
  if (definition && definition.fields) {
    for (let i = 0; i < definition.fields.length; i++) {
      const field = definition.fields[i];
      const { name, type } = field;
//////////////console.error("COMPUTING FIELD SIZE: ", field);
      if (fields.hasOwnProperty(name)) {
        switch (BaseType[type]) {
			case BaseType.BYTE:
          case BaseType.ENUM:
          case BaseType.UINT8:
          case BaseType.UINT8Z:
          case BaseType.SINT8:
            dataSize += 1;
            break;
          case BaseType.UINT16:
          case BaseType.UINT16Z:
          case BaseType.SINT16:
            dataSize += 2;
            break;
          case BaseType.UINT32:
          case BaseType.UINT32Z:
          case BaseType.SINT32:
          case BaseType.FLOAT32:
            dataSize += 4;
            break;
          case BaseType.FLOAT64:
            dataSize += 8;
            break;
          case BaseType.STRING:
            const stringLength = field.size - 1;
            dataSize += stringLength;
            break;
          case BaseType.UINT8_ARRAY:
          case BaseType.UINT8Z_ARRAY:
            dataSize += field.size;
            break;
          case BaseType.UINT16_ARRAY:
          case BaseType.UINT16Z_ARRAY:
            dataSize += field.size * 2;
            break;
          case BaseType.UINT32_ARRAY:
          case BaseType.UINT32Z_ARRAY:
          case BaseType.SINT32_ARRAY:
            dataSize += field.size * 4;
            break;
          default:
            throw new Error(`Unsupported field type: ${type}`);
        }
      }
    }
  }
  if (definition && definition.developerFields) {
    for (let i = 0; i < definition.developerFields.length; i++) {
      const field = definition.developerFields[i];
      const { name, type } = field;
//////////////console.error("CALCULATING DEVELOPER FIELD LENGTH: ", field);
      if (fields.hasOwnProperty(name)) {
        switch (BaseType[type]) {
          case BaseType.ENUM:
          case BaseType.UINT8:
          case BaseType.UINT8Z:
          case BaseType.SINT8:
            dataSize += 1;
            break;
          case BaseType.UINT16:
          case BaseType.UINT16Z:
          case BaseType.SINT16:
            dataSize += 2;
            break;
          case BaseType.UINT32:
          case BaseType.UINT32Z:
          case BaseType.SINT32:
          case BaseType.FLOAT32:
            dataSize += 4;
            break;
          case BaseType.FLOAT64:
            dataSize += 8;
            break;
          case BaseType.STRING:
            const stringLength = field.size - 1;
            dataSize += stringLength;
            break;
          case BaseType.UINT8_ARRAY:
          case BaseType.UINT8Z_ARRAY:
            dataSize += field.size;
            break;
          case BaseType.UINT16_ARRAY:
          case BaseType.UINT16Z_ARRAY:
            dataSize += field.size * 2;
            break;
          case BaseType.UINT32_ARRAY:
          case BaseType.UINT32Z_ARRAY:
          case BaseType.SINT32_ARRAY:
            dataSize += field.size * 4;
            break;
          default:
            throw new Error(`Unsupported field type: ${type}`);
        }
      }
    }
  }
  //////////////console.warn("DATA LENGTH: ", dataSize);
  
  return dataSize;
}
findDefinition(messageType) {
  const definition = Profile.messages[messageType];
  //////////////console.warn(definition);
  if (definition) {
    return definition;
  } else {
    //////////////console.error(`Definition for message type ${messageType} not found.`);
    return null;
  }
}
static PrintBits(num) {
  return num.toString(2).padStart(8, '0');
}
readValue(dataView, field, currentOffset, arch) {
let fieldType = field.type;
  switch (fieldType) {
	  case BaseType.BYTE:
    case BaseType.ENUM:
    case BaseType.UINT8:
    case BaseType.UINT8Z:
      return dataView.getUint8(currentOffset);
    case BaseType.SINT8:
      return dataView.getInt8(currentOffset);
    case BaseType.UINT16:
    case BaseType.UINT16Z:
      return dataView.getUint16(currentOffset, arch === 'little');
    case BaseType.SINT16:
      return dataView.getInt16(currentOffset, arch === 'little');
    case BaseType.UINT32:
    case BaseType.UINT32Z:
      return dataView.getUint32(currentOffset, arch === 'little');
    case BaseType.SINT32:
      return dataView.getInt32(currentOffset, arch === 'little');
    case BaseType.STRING:
      return String.fromCharCode.apply(null, new Uint8Array(dataView.buffer, currentOffset, field.size - 1));
    case BaseType.FLOAT32:
      return dataView.getFloat32(currentOffset, arch === 'little');
    case BaseType.FLOAT64:
      return dataView.getFloat64(currentOffset, arch === 'little');
    case BaseType.UINT8Z_ARRAY:
      return Array.from(new Uint8Array(dataView.buffer, currentOffset, field.size));
    case BaseType.UINT16_ARRAY:
    case BaseType.UINT16Z_ARRAY:
      return Array.from(new Uint16Array(dataView.buffer, currentOffset, field.size / 2));
    case BaseType.UINT32_ARRAY:
    case BaseType.UINT32Z_ARRAY:
      return Array.from(new Uint32Array(dataView.buffer, currentOffset, field.size / 4));
    case BaseType.SINT32_ARRAY:
      return Array.from(new Int32Array(dataView.buffer, currentOffset, field.size / 4));
    default:
      throw new Error(`Unsupported field type: ${fieldType}`);
  }
}

parseDataFields(dataView, offset, definition) {
	
  let fields;
  let arch = definition.arch;
  //////////////console.error(arch);
  const parsedFields = {};
  const parsedBytes = {};
  let currentOffset = offset;
  for (let k = 0; k < 2; k++) {
    if (k === 0) {
      fields = definition.fields;
    } else {
      fields = definition.developerFields;
    }
    if (fields) {
      if (!Array.isArray(fields)) {
        fields = Object.values(fields);
        for (let i = 0; i < fields.length; i++) {
          fields[i].fieldDefNum = fields[i].num;
          if (!FieldTypeToBaseType[fields[i].type]) {
            //fields[i].type =2;
          } else {
            fields[i].type = FieldTypeToBaseType[fields[i].type];
          }
        }
      }

        for (let i = 0; i < fields.length; i++) {
const field = fields[i];
const { name, type } = field;
const fieldType = BaseType[type];
const scale = field.scale || 1;
const byteArray = (length) => new Uint8Array(dataView.buffer.slice(currentOffset, currentOffset + length));

parsedFields[name] = this.readValue(dataView, field, currentOffset, arch) / scale;

if (!parsedFields.units) {
  parsedFields.units = {};
}

parsedFields.units[name] = field.units || "";
    parsedBytes[name] = byteArray(field.size);
//////////////console.error(fieldType, field, parsedFields[name]);
    switch (fieldType) {
		case BaseType.BYTE:
      case BaseType.ENUM:
      case BaseType.UINT8:
      case BaseType.UINT8Z:
      case BaseType.SINT8:
        currentOffset += 1;
        break;
      case BaseType.UINT16:
      case BaseType.UINT16Z:
      case BaseType.SINT16:
        currentOffset += 2;
        break;
      case BaseType.UINT32:
      case BaseType.UINT32Z:
      case BaseType.SINT32:
        currentOffset += 4;
        break;
      case BaseType.STRING:
        currentOffset += field.size - 1;
        break;
      case BaseType.FLOAT32:
        currentOffset += 4;
        break;
      case BaseType.FLOAT64:
        currentOffset += 8;
        break;
      case BaseType.UINT8Z_ARRAY:
      case BaseType.UINT16_ARRAY:
      case BaseType.UINT16Z_ARRAY:
      case BaseType.UINT32_ARRAY:
      case BaseType.UINT32Z_ARRAY:
      case BaseType.SINT32_ARRAY:
        currentOffset += field.size;
        break;
      default:
        throw new Error(`Unsupported field type: ${fieldType}`);
    }
  }
    }
  }
  return { parsedFields, parsedBytes };
}

async parseBody(buffer, definitions) {
  if (!definitions) {
    definitions = {};
  }
  const dataView = new DataView(buffer);
  
 
  const messages = [];
  let index = 0;
  

  const workerTask = async (workerIndex) => {
 if(index < buffer.byteLength){
      // Parse the record header
      const headerByte = dataView.getUint8(index);
      
      const isCompressed = (headerByte >> 7) & 1;
      const isDefinition = (headerByte >> 6) & 1;
      const hasDeveloperFields = (headerByte >> 5) & 1;
      const localMessageType = headerByte & 0x0F;

      index++;

      let message;
      if (isDefinition) {
        const numFields = dataView.getUint8(index + 4);
        let devFieldLength = 0;
        if (hasDeveloperFields) {
          const numDevFields = dataView.getUint8(index + 4 + numFields * 3 + 1);
          devFieldLength = numDevFields * 3 + 1;
        }
        const definitionBytes = buffer.slice(index, index + 4 + numFields * 3 + devFieldLength + 1);
        const parsedDefinition = this.parseDefinition(definitionBytes, headerByte);
        definitions[localMessageType] = parsedDefinition;
        index += definitionBytes.byteLength;
        message = { type: 'definition', localMessageType, fields: parsedDefinition.fields };

      } else if (isCompressed) {
        const timeOffset = headerByte >> 2 & 0x1F;
        const prevTimestamp = messages[messages.length - 1]?.timestamp || 0;
        const timestamp = (prevTimestamp & 0xFFFFFFF8) | ((prevTimestamp & 0x7) + timeOffset) & 0x1F;
        const localMessageType = headerByte & 0x0F;
        const definition = definitions[localMessageType];
        const numElements = definition ? definition.numFields : 0;
      //  if (definition) {
          index += numElements;
          message = { type: 'compressed', localMessageType, timestamp };
       // }

      } else {
        let definition = definitions[localMessageType];
        if (definition) {
          const parsing = this.parseDataFields(dataView, index, definition);
          let fields = parsing.parsedFields;
          fields.byteArrays = parsing.parsedBytes;
          index += this.computeDataSize(fields, definition, definition.arch);
          message = { type: 'data', localMessageType, fields };
        } else {
          // Definition message not found, log a warning and continue parsing
        }
      }
      ; //cancel out worker incrementor
          return message;
    }


  };

  const worker = new MyWorker(
    this.chainlink,
    workerTask,
    "Parsing Body",
    buffer.byteLength,
    0,
    this.parentID
  );

  await worker.run((i)=> {return index;});
  //////console.log(worker.result);
  return worker.result;

}
 calculateChecksum(buffer) {
  const crcCalculator = new CrcCalculator();
  const crc = crcCalculator.addBytes(new Uint8Array(buffer), 0, buffer.byteLength);
  return crc;
}

verifyCRC(buffer, crc) {
  const calculatedCRC = this.calculateChecksum(buffer);
  const returnVal = calculatedCRC === crc;
  if(!returnVal){console.warn(buffer, "crc mismatch", crc, ":", calculatedCRC);}
  else{
	  console.warn("crc match");
  }
  return returnVal;
}
parseHeader(buffer) {
  const header = {};
  const view = new DataView(buffer);

  // Byte 0: header size (minimum is 12)
  header.headerSize = view.getUint8(0);

  // Byte 1: protocol version
  header.protocolVersion = view.getUint8(1);

  // Bytes 2-3: profile version
  header.profileVersion = view.getUint16(2, true);

  // Bytes 4-7: data size (excluding header and CRC)
  header.dataSize = view.getUint32(4, true);

  // Bytes 8-11: "FIT " (file identifier)
  header.fileIdentifier = String.fromCharCode(
    view.getUint8(8),
    view.getUint8(9),
    view.getUint8(10),
    view.getUint8(11)
  );
  if(header.headerSize >= 14){
  // Bytes 12-13: header CRC
  header.headerCRC = view.getUint16(12, true);

  // Verify header CRC
  const headerWithoutCRC = buffer.slice(0, 12);// + buffer.slice(14, header.headerSize - 2);
  header.isValid = this.verifyCRC(headerWithoutCRC, header.headerCRC);
}
  return header;
}


parseDefinition(buffer, headerByte) {
  const dataView = new DataView(buffer);
  const localMessageType = headerByte & 0x0F;
  let arch = dataView.getUint8(1) ? 'big' : 'little';
  //////////////console.error(dataView.getUint8(1), ":", arch);
  //arch = 'big';
  const globalMessageNumber = dataView.getUint16(2, arch === 'little'); //remember you changed this and it suddenly worked
  ////////////////console.error(buffer);
  
  
  const numFields = dataView.getUint8(4);
  const fields = [];
  let index = 5;
  for (let i = 0; i < numFields; i++) {
    const fieldDefNum = dataView.getUint8(index++);
    const size = dataView.getUint8(index++);
    const type = dataView.getUint8(index++);
    const [fieldName, scale, units] = this.findFieldData(globalMessageNumber, fieldDefNum);
    const field = { fieldDefNum, size, type, name: fieldName, scale, units };
    fields.push(field);
  }

  let developerDataSize = 0;
  let developerFields = [];
  if ((headerByte >> 5) & 1) {
    const numDevFields = dataView.getUint8(index++);
    developerFields = [];

    for (let i = 0; i < numDevFields; i++) {
      const fieldDefNum = dataView.getUint8(index++);
      const size = dataView.getUint8(index++);
      const devDataIndex = dataView.getUint8(index++);
      this.developers[devDataIndex] = {fieldDefNum, size};
     // const [fieldName, scale, units] = this.findDevFieldData(devDataIndex, fieldDefNum);
      const field = { fieldDefNum, size, devDataIndex,};// name: fieldName, scale, units };
      developerFields.push(field);
    }

    
  }

  const messageSize = index;
  //const devData = developerDataSize > 0 ? buffer.slice(index, index + developerDataSize) : null;

  const definition = {
	arch,
    localMessageType,
    globalMessageNumber,
    fields,
    numDeveloperFields: developerFields.length,
    developerFields,
    messageSize,
    developerDataSize,
  };
//////////////console.warn("We have defined:", definition);
    this.localMessageDefinitions[localMessageType] = definition;
  return definition;
}
 findDevFieldData(devDataIndex, fieldNum){
	const fieldName = this.developers[devDataIndex].fields[fieldNum].name;
	const scale =  this.developers[devDataIndex].fields[fieldNum].scale;
	const units = this.developers[devDataIndex].fields[fieldNum].units;
	return [fieldName, scale, units];
}
 findFieldData(globalMessageNumber, fieldNum) {
  //////////////console.warn(globalMessageNumber);
  const fields = Profile.messages[globalMessageNumber].fields;
  const fieldName = fields[fieldNum].name;
	const scale = fields[fieldNum].scale; 
	const units = fields[fieldNum].units;
	return [fieldName, scale, units];
}
async parseFitFile(file) {
  console.error("parsing fit file: ", file);
  
  const blob = await FitFile.getFileBlob(file);
console.warn("the blob is obtained.");
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async (event) => {
      const fileContent = event.target.result;
      console.warn("file has been loaded");
      // Parse the file content here
      const buffer = fileContent;//new ArrayBuffer(fileContent.length);
      const uint8View = new Uint8Array(buffer);
      for (let i = 0; i < fileContent.length; i++) {
        uint8View[i] = fileContent.charCodeAt(i);
      }

      // Parse the header
      const header = this.parseHeader(buffer);
      console.error(header);
      // Parse the data messages in the body
      let bodyBuffer;
      if (header.dataSize + header.headerSize === buffer.byteLength) {
        // No extra CRC bytes at the end of the file
        bodyBuffer = buffer.slice(header.headerSize);
      } else if (header.dataSize + header.headerSize + 2 === buffer.byteLength) {
        // Possible CRC bytes at the end of the file, check if they match the calculated CRC
        const crcPosition = buffer.byteLength - 2;
        //const calculatedCRC = this.calculateChecksum(buffer.slice(header.headerSize, crcPosition));
        const fileCRC = new DataView(buffer).getUint16(crcPosition, true);
        const bodyValid = this.verifyCRC(buffer.slice(0, buffer.byteLength-2), fileCRC);
        if (bodyValid) {
          // CRC check passed, remove the CRC bytes from the body buffer
          bodyBuffer = buffer.slice(header.headerSize, crcPosition);
        } else {
          // CRC check failed, reject the Promise
         // console.error("INVALID FILE CRC ", calculatedCRC, ":", fileCRC);
          reject(new Error('Invalid file CRC'));
          return;
        }
      } else {
        // File size does not match header data size, reject the Promise
        console.error("INVALID FILE SIZE");
        reject(new Error('Invalid file size'));
        return;
      }

      const body = await this.parseBody(bodyBuffer);
      console.warn(body);
      // Resolve the Promise with the parsed body
      resolve(body);
    };

    reader.onerror = function() {
      reject(new Error('Error reading the file'));
    };

    reader.readAsArrayBuffer(blob);
  });
}

static async getFileBlob(file) {
	try{
		let returnval;
  if (file instanceof Blob) {
    returnval = file;
  } else if (file instanceof File) {
    returnval = file.slice(0, file.size, file.type);
  } else if (typeof file === 'string') {
	  console.warn("obtaining file: ", file);
    const response = await fetch(file);
    returnval = await response.blob();
  } else {
    throw new Error('Invalid file type');
  }
  console.warn("verifying blob...");
  console.warn("blob status: ", FitFile.verifyBlob(returnval));
  return returnval;
  }catch(error){
	  console.error(error);
  }
}

static  verifyBlob(blob) {
  if (!blob) {
    console.error("Blob is null or undefined.");
    return false;
  }
  
  if (blob.type !== "application/octet-stream") {
    console.error("Blob has unexpected MIME type: " + blob.type);
    return false;
  }
  
  if (blob.size === 0) {
    console.error("Blob has zero size.");
    return false;
  }
  
  const reader = new FileReader();
  reader.onload = function(event) {
    const arrayBuffer = event.target.result;
    console.log("Blob contents:", arrayBuffer);
  };
  reader.readAsArrayBuffer(blob);
  
  return true;
}

encodeHeader(header) {
  const buffer = new ArrayBuffer(header.headerSize);
  const view = new DataView(buffer);

  // Byte 0: header size
  view.setUint8(0, header.headerSize);

  // Byte 1: protocol version
  view.setUint8(1, header.protocolVersion);

  // Bytes 2-3: profile version
  view.setUint16(2, header.profileVersion, true);

  // Bytes 4-7: data size
  view.setUint32(4, header.dataSize, true);

  // Bytes 8-11: "FIT " (file identifier)
  for (let i = 0; i < 4; i++) {
    view.setUint8(8 + i, header.fileIdentifier.charCodeAt(i));
  }

  // Calculate and set header CRC
  if (header.headerSize >= 14) {
    const headerWithoutCRC = buffer.slice(0, 12);
    const headerCRC = this.calculateChecksum(headerWithoutCRC);
    view.setUint16(12, headerCRC, true);
  }

  return buffer;
}
async encodeFitFile(headerFields, messageObjects) {
const encodedBody = this.encodeBody(messageObjects);
const encodedHeader = this.encodeHeader({
headerSize: 14,
protocolVersion: 16,
profileVersion: 209,
dataSize: encodedBody.byteLength,
fileIdentifier: "FIT ",
});

// Combine header and body
const buffer = new ArrayBuffer(encodedHeader.byteLength + encodedBody.byteLength + 2);
const uint8View = new Uint8Array(buffer);
uint8View.set(new Uint8Array(encodedHeader), 0);
uint8View.set(new Uint8Array(encodedBody), encodedHeader.byteLength);

// Calculate and append the CRC
const crc = this.calculateChecksum(buffer.slice(0, buffer.byteLength - 2));
const dataView = new DataView(buffer);
dataView.setUint16(buffer.byteLength - 2, crc, true);

return buffer;
}
encodeBody(messages, definitions) {
  let messageBuffer = new ArrayBuffer(0);
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    const messageType = message.localMessageType;
    const definition = definitions[messageType];
    if (!definition) {
      console.warn(`Definition for message type ${messageType} not found.`);
      continue;
    }

    let messageBytes = new ArrayBuffer(0);
    if (message.type === "definition") {
      const numFields = message.fields.length;
      const numDevFields = message.developerFields?.length || 0;
      const hasDeveloperFields = numDevFields > 0;
      const buffer = new ArrayBuffer(4 + numFields * 3 + numDevFields * 3 + 1);
      const view = new DataView(buffer);
      const globalMessageNumber = definition.globalMessageNumber;
      const reserved = 0;

      view.setUint8(0, 0x40 | messageType);
      view.setUint8(1, definition.arch === "big" ? 1 : 0);
      view.setUint16(2, globalMessageNumber, definition.arch === "little");
      view.setUint8(4, numFields);

      let offset = 5;
      for (let j = 0; j < numFields; j++) {
        const field = message.fields[j];
        view.setUint8(offset, field.fieldDefNum);
        view.setUint8(offset + 1, field.size);
        view.setUint8(offset + 2, field.type);
        offset += 3;
      }

      if (hasDeveloperFields) {
        view.setUint8(offset, numDevFields);
        offset++;
        for (let j = 0; j < numDevFields; j++) {
          const field = message.developerFields[j];
          view.setUint8(offset, field.fieldDefNum);
          view.setUint8(offset + 1, field.size);
          view.setUint8(offset + 2, field.devDataIndex);
          offset += 3;
        }
      }

      messageBytes = buffer;
    } else if (message.type === "compressed") {
      const timeOffset = message.timestamp & 0x1f;
      const buffer = new ArrayBuffer(1);
      const view = new DataView(buffer);
      view.setUint8(0, 0x80 | messageType | (timeOffset << 2));
      messageBytes = buffer;
    } else if (message.type === "data") {
      const fields = message.fields;
      const arch = definition.arch;
      const buffer = new ArrayBuffer(this.computeDataSize(fields, definition, arch));
      const view = new DataView(buffer);

      let offset = 0;
      for (let j = 0; j < definition.fields.length; j++) {
        const field = definition.fields[j];
        const { name, type } = field;
        const fieldType = BaseType[type];
        const scale = field.scale || 1;
        let value = fields[name] * scale;

        if (fieldType === BaseType.STRING) {
          value = value.split("").map((c) => c.charCodeAt(0));
        }

        if (fieldType === BaseType.UINT8Z_ARRAY) {
          value = new Uint8Array(value);
        }

        if (fieldType === BaseType.UINT8_ARRAY) {
	  value = new Int8Array(value);
	}
	if (fieldType === BaseType.UINT16_ARRAY) {
      value = new Uint16Array(value);
    }

    if (fieldType === BaseType.UINT32_ARRAY) {
      value = new Uint32Array(value);
    }

    if (fieldType === BaseType.SINT8_ARRAY) {
      value = new Int8Array(value);
    }

    if (fieldType === BaseType.SINT16_ARRAY) {
      value = new Int16Array(value);
    }

    if (fieldType === BaseType.SINT32_ARRAY) {
      value = new Int32Array(value);
    }

    if (fieldType === BaseType.FLOAT32_ARRAY) {
      value = new Float32Array(value);
    }

    if (fieldType === BaseType.FLOAT64_ARRAY) {
      value = new Float64Array(value);
    }

    if (fieldType === BaseType.UINT8Z) {
      value = value >>> 0;
    }

    if (fieldType === BaseType.SINT8) {
      value = value << 24 >> 24;
    }

    if (fieldType === BaseType.UINT8) {
      value = value >>> 0;
    }

    if (fieldType === BaseType.SINT16) {
      value = value << 16 >> 16;
    }

    if (fieldType === BaseType.UINT16) {
      value = value >>> 0;
    }

    if (fieldType === BaseType.SINT32) {
      value |= 0;
    }

    if (fieldType === BaseType.UINT32) {
      value >>>= 0;
    }

    if (fieldType === BaseType.FLOAT32) {
      view.setFloat32(offset, value, arch === "little");
      offset += 4;
      continue;
    }

    if (fieldType === BaseType.FLOAT64) {
      view.setFloat64(offset, value, arch === "little");
      offset += 8;
      continue;
    }

    const byteSize = field.size;
    const isVariable = byteSize === 0xff;
    const isArray = fieldType >= BaseType.UINT8Z_ARRAY;

    if (isVariable) {
      byteSize = value.length;
      view.setUint8(offset, byteSize);
      offset++;
    }

    if (isArray) {
      byteSize *= value.length;
    }

    if (byteSize > 0) {
      if (isArray) {
        const byteArray = new Uint8Array(buffer, offset, byteSize);
        for (let k = 0; k < value.length; k++) {
          byteArray.set(new Uint8Array(value[k].buffer), k * field.size);
        }
      } else if (fieldType === BaseType.STRING) {
        const byteArray = new Uint8Array(buffer, offset, byteSize);
        for (let k = 0; k < value.length; k++) {
          byteArray.set(value[k], k);
        }
      } else {
        view[`set${fieldType}`](offset, value, arch === "little");
      }

      offset += byteSize;
    }
  }

  messageBytes = buffer;
}

messageBuffer = this.concatArrayBuffers(messageBuffer, messageBytes);
}

return messageBuffer;
}

	}