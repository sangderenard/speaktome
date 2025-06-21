class GCMEncodedMessage {
  constructor(dataBlock, keypair, hashedPassword) {
    this.dataBlock = dataBlock;
    this.messageNumber = 0;
    this.gcmInitializationParams = null;
    this.keypair = keypair;
    this.hashedPassword = hashedPassword;
  }

  async encrypt() {
    const { iv, tagLength } = this.gcmInitializationParams;
    const message = new Uint8Array([...this.dataBlock, ...this.messageNumber.toString().padStart(16, '0')].map(Number));
    const derivedKey = await ServerCommunication.deriveGcmKey(iv, this.hashedPassword);
    const encryptedData = await ServerCommunication.encryptWithGCM(derivedKey, message);
    const tag = encryptedData.slice(encryptedData.length - tagLength / 8);
    return new Uint8Array([...encryptedData, ...tag].map(Number));
  }

  async decrypt(key, encryptedMessage) {
    const { iv, tagLength } = this.gcmInitializationParams;
    const message = await ServerCommunication.decryptWithGCM(key, {
      iv,
      tag: encryptedMessage.slice(encryptedMessage.length - tagLength / 8)
    }, encryptedMessage.slice(0, encryptedMessage.length - tagLength / 8));
    const messageNumberString = String(message.slice(-16));
    const messageNumber = Number.parseInt(messageNumberString, 10);
    if (Number.isNaN(messageNumber)) {
      throw new Error(`Failed to parse message number: ${messageNumberString}`);
    }
    this.messageNumber = messageNumber;
    this.previousTag = encryptedMessage.slice(encryptedMessage.length - tagLength / 8);
    return message.slice(0, -16);
  }

  encodeAsUriComponent() {
    return encodeURIComponent(btoa(new Uint8Array(this.dataBlock)));
  }

  static decodeFromUriComponent(uriComponent, dataBlockSize) {
    const decoded = atob(decodeURIComponent(uriComponent));
    const dataBlock = new Uint8Array([...decoded.padEnd(dataBlockSize * 2, '0')].map(Number)).slice(0, dataBlockSize);
    return new GCMEncodedMessage(dataBlock);
  }

  async updateGcmInitializationParams(privateKey) {
    const { iv } = this.gcmInitializationParams;
    const randomData = window.crypto.getRandomValues(new Uint8Array(16 * 16));
    const encryptedData = await ServerCommunication.encryptWithGCM(await ServerCommunication.deriveGcmKey(iv, privateKey), randomData);
    const encryptedDataUri = encodeURIComponent(btoa(new Uint8Array(encryptedData)));

    const gcmConfig = {
      iv,
      additionalData: null,
      tagLength: 128
    };
    const encryptedConfig = await ServerCommunication.encryptWithRSA(publicKey, JSON.stringify(gcmConfig));
    const encryptedConfigUri = encodeURIComponent(btoa(new Uint8Array(encryptedConfig)));

    const url = `https://bike.dogbite.me/cgi-bin/privkey.php?hashedpasscode=${hashedPassword}&encryptedData=${encryptedDataUri}&encryptedConfig=${encryptedConfigUri}`;
    // Send the URL request here
    this.messageNumber = 0;
    this.gcmInitializationParams = gcmConfig;
  }
}




class ServerCommunication{

static async buildDirectorySelector(folderLocation, selectElement, serverResponse){
      
      try{
		  selectElement.innerHTML = "";
 const response = await fetch("https://bike.dogbite.me/cgi-bin/directoryListing.php?directory_path=" + encodeURIComponent(folderLocation));
  const files = await response.json();
  
      files.forEach(async (file) => {
    try {
        const optionElement = document.createElement("option");
        optionElement.value = await InputControl.getFileObject("https://bike.dogbite.me/cgi-bin/"+encodeURIComponent(folderLocation)+"/"+encodeURIComponent(file.name));
        optionElement.textContent = file.name;
        selectElement.appendChild(optionElement);
    } catch (error) {
        console.log("Error adding option for file:", file.name, "Error:", error);
    }
      });
      serverResponse = files;
      return response;
      
      }catch(error){
		  console.log("error loading: ", folderLocation, ": ", error);
	  }
}

static async uploadFile(file, combinedData, serverResponse) {
  // Create a new FormData object and append the file and combined encrypted data
  const formData = new FormData();
  formData.append("file", file);
  formData.append("payload", combinedData);

  // Send a POST request to the PHP script URL using fetch
  const response = await fetch("https://bike.dogbite.me/cgi-bin/fileUpload.php", {
    method: "POST",
    body: formData
  });
serverResponse.value = await response.text();
  // Check if the response status is OK and return true if successful
  if (response.status === 200) {
    return true;
  } else {
    return false;
  }
}
static async getMaxDataSize(key){
	return 190;
	
	//return to this at a later date and fix it.
	
	
  const padding = 42;
  const hashAlgorithm = 'SHA-256';
  const keyAlgorithm = {
    name: 'RSA-OAEP',
    hash: hashAlgorithm,
  };
  const maxChunkSize = Math.floor((await window.crypto.subtle.exportKey('jwk', key)).n.length / 8) - padding;
  const encoder = new TextEncoder();
  const testString = 'x'.repeat(maxChunkSize);
  const testData = encoder.encode(testString);
  let encodedData = null;
  let i = maxChunkSize;
  while (encodedData === null && i > 0) {
    try {
      const partialData = testData.subarray(0, i);
      const encodedPartialData = await window.crypto.subtle.encrypt(keyAlgorithm, key, partialData);
      encodedData = encodedPartialData;
    } catch (e) {
      i--;
    }
  }
  if (encodedData === null) {
    throw new Error('Could not determine maximum data size');
  }
  return i - padding;
};

















static async initializeAES(password, salt = null){
try {
		let salt, key, hashedPassword, iv;
		hashedPassword = CryptoJS.SHA256(password).toString();
		if(!salt){
			salt = CryptoJS.lib.WordArray.random(16);
		}
		const iterations = 10000;
		key = CryptoJS.PBKDF2(hashedPassword, salt, {hasher: CryptoJS.algo.SHA256,
	  		keySize: 256/32,
	  		iterations: iterations
		});
		iv = CryptoJS.lib.WordArray.random(16);		
		return {salt, key, hashedPassword, iv};
	} catch (error) {
  		console.error(error);
	}
}
static async deriveGcmKey(iv, password) {
  try {
    const { salt, key } = await ServerCommunication.initializeAES(password);
    const gcmKey = await window.crypto.subtle.importKey('raw', key, { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']);
    const derivedKey = await window.crypto.subtle.deriveKey(
      {
        name: 'PBKDF2',
        salt: salt,
        iterations: 10000,
        hash: 'SHA-256'
      },
      gcmKey,
      { name: 'AES-GCM', length: 256 },
      true,
      ['encrypt', 'decrypt']
    );
    return derivedKey;
  } catch (error) {
    console.error('error deriving GCM key', error);
  }
}

static async encryptWithGCM(key, data, crc) {
  const algorithm = { name: 'AES-GCM', iv: key.iv, tagLength: 128 };
  const dataWithCrc = new Uint8Array(data.length + crc.length);
  dataWithCrc.set(data);
  dataWithCrc.set(crc, data.length);
  return await crypto.subtle.encrypt(algorithm, key, dataWithCrc);
}

static async encryptWithRSA(publicKey, data) {
  return await window.crypto.subtle.encrypt({
    name: 'RSA-OAEP'
  }, publicKey, new TextEncoder().encode(data));
}

static async decryptWithGCM(key, config) {
  const algorithm = { name: 'AES-GCM', iv: config.iv, tagLength: 128, tag: config.tag };
  return await crypto.subtle.decrypt(algorithm, key, new Uint8Array(key));
}


static async getServerKeys(hashedPassword) {
  let privateKey, publicKey, chunkSize;
  try {
    const publicKeyUrl = `https://bike.dogbite.me/cgi-bin/pubkey.php?hashedpasscode=${hashedPassword}`;
    const publicKeyBuffer = await ServerCommunication.pemFileToKeyBuffer(publicKeyUrl);
    console.warn(`IMPORTING PUBLIC KEY FROM: ${publicKeyUrl}`);
    const publicKeyType = 'spki'; // use spki for public keys
    publicKey = await window.crypto.subtle.importKey(publicKeyType, publicKeyBuffer, {
      name: 'RSA-OAEP',
      hash: 'SHA-512'
    }, true, ['encrypt']);
    chunkSize = await ServerCommunication.getMaxDataSize(publicKey);
    console.log('Public Key Imported');

    // Generate a random 16x16 byte array for the IV and a separate 16 byte array for the data
    const randomIv = window.crypto.getRandomValues(new Uint8Array(16));
    const randomData = window.crypto.getRandomValues(new Uint8Array(16 * 16));
    const crc = new CrcCalculator().addBytes(randomData, 0, randomData.byteLength);

    // Derive the GCM key from the previous tag and the new IV
    const previousTag = null; // set to the previous tag
    const gcmKey = await ServerCommunication.deriveGcmKey(previousTag, randomIv, hashedPassword);

    // Encrypt the CRC and random data block using GCM and encode the result as a URI component
    const encryptedData = await ServerCommunication.encryptWithGCM(gcmKey, randomData, crc);
    const encryptedDataUri = encodeURIComponent(btoa(new Uint8Array(encryptedData)));

    // Encrypt the GCM initialization details using the server's public key and encode the result as a separate URI component
    const gcmConfig = {
      iv: randomIv,
      additionalData: null,
      tagLength: 128
    };
    const encryptedConfig = await ServerCommunication.encryptWithRSA(publicKey, JSON.stringify(gcmConfig));
    const encryptedConfigUri = encodeURIComponent(btoa(new Uint8Array(encryptedConfig)));

    // Send the encoded encrypted data and the encrypted GCM initialization details to the server in the URL
    const privateKeyUrl = `https://bike.dogbite.me/cgi-bin/privkey.php?hashedpasscode=${hashedPassword}&encryptedData=${encryptedDataUri}&encryptedConfig=${encryptedConfigUri}`;
    console.warn(`IMPORTING PRIVATE KEY FROM: ${privateKeyUrl}`);

    // Decrypt the response using GCM initialization details derived from the first 16 bytes of the random data
    const decryptedData = await ServerCommunication.decryptWithGCM(privateKeyBuffer, {
      iv: randomIv,
      tag: null // set to the tag from the response
    });
    const privateKeyType = 'pkcs8'; // use pkcs8 for private keys
    privateKey = await window.crypto.subtle.importKey(privateKeyType, decryptedData, {
      name: 'RSA-OAEP',
      hash: 'SHA-512'
    }, false, ['decrypt']);
    console.log('Private Key Imported');

  } catch (error) {
    console.error('error obtaining key', error);
  }
  return { publicKey, privateKey, chunkSize };
}
}