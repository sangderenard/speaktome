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
static async pemFileToKeyBuffer(url, decryptionKey) {
  try {
    const response = await fetch(url);
    const text = await response.text();

    // Extract the PEM-encoded key from the file
    const match = text.match(/-----BEGIN ((?:ENCRYPTED )?(?:PUBLIC|PRIVATE) KEY)-----\n([\s\S]*)\n-----END \1-----/);
    if (!match) {
      throw new Error('Invalid PEM file format');
    }
    const keyString = match[2].replace(/\s+/g, '');
    console.warn("I received this pem file contents: ", keyString);

// Convert the base64-encoded key to a Uint8Array buffer
let keyBuffer = Uint8Array.from(atob(keyString), c => c.charCodeAt(0)).buffer;

    // If the key is a private key and an encryption key is provided, decrypt the key
    if (url.includes('privkey.php') && decryptionKey) {

      // Extract the length of the RSA-encrypted GCM parameters
      const dataView = new DataView(keyBuffer, 0, 2);
      const gcmLength = dataView.getUint16(0, false);
      console.warn("The GCM Parameter data length is: ", gcmLength);

      // Extract the RSA-encrypted GCM parameters
      const encryptedGCMParams = keyBuffer.slice(2, 2 + gcmLength);

		let decryptedGCMParams;
		try{
      // Decrypt the GCM parameters using the decryption key
       decryptedGCMParams = await crypto.subtle.decrypt(
        { name: "RSA-OAEP", hash: {name: "SHA-512"} },
        decryptionKey,
        encryptedGCMParams
      );
      }catch(error){
		console.error("There was a problem decrypting.");
		console.warn("The data: ", encryptedGCMParams);
		console.warn("The key: ", decryptionKey);  
		  
	  }
      
      

      // Extract the nonce and tag from the decrypted GCM parameters
      const nonce = decryptedGCMParams.slice(0, 12);
      const tag = decryptedGCMParams.slice(12);

      // Decrypt the key data using AES-GCM with the nonce and tag
      const encryptedKeyData = keyBuffer.slice(2 + gcmLength);
      const decryptedKeyData = await crypto.subtle.decrypt(
        { name: "AES-GCM", iv: nonce, tagLength: 128 },
        encryptedKeyData,
        { name: "GCM", tagLength: 128, tag: tag }
      );

      // The decrypted key data is in decryptedKeyData
    }

    // Return the key buffer
    return keyBuffer;

  } catch (error) {
    console.error("Error obtaining key", error);
  }
}
static async makeRSAKeys(){
	let keyPair;
	try {
    // Generate a new RSA key pair for the client
    keyPair = await window.crypto.subtle.generateKey({
      name: "RSA-OAEP",
      modulusLength: 4096,
      publicExponent: new Uint8Array([0x01, 0x00, 0x01]),
      hash: "SHA-512"
    }, true, ["encrypt", "decrypt"]);
  } catch (error) {
    console.error(error);
  }
  return keyPair;
} 
static async testEncryption(hashedPassword) {
	
  try {
    // Generate a new RSA key pair for testing locally
    const localKeyPair = await ServerCommunication.makeRSAKeys();

    // Generate a random message to encrypt
    const message = new Uint8Array(32);
    window.crypto.getRandomValues(message);

    // Encrypt the message with the local public key
    const encrypted = await window.crypto.subtle.encrypt({ name: 'RSA-OAEP' }, localKeyPair.publicKey, message);

    // Decrypt the message with the local private key
    const decrypted = new Uint8Array(await window.crypto.subtle.decrypt({ name: 'RSA-OAEP' }, localKeyPair.privateKey, encrypted));

    // Compare the original message and the decrypted message to ensure they are the same
    let isEqual = true;
    for (let i = 0; i < message.length; i++) {
      if (message[i] !== decrypted[i]) {
        isEqual = false;
        break;
      }
    }
console.log(message, ":", decrypted);
    console.log('Local encryption/decryption test passed:', isEqual);
try {
  // Export the public key in SPKI format
  const publicKey = await crypto.subtle.exportKey('spki', localKeyPair.publicKey);

  // Encode the key as base64
  const publicKeyEncoded = btoa(String.fromCharCode(...new Uint8Array(publicKey)));

  // URI-encode the base64-encoded key
  const publicKeyUriEncoded = encodeURIComponent(publicKeyEncoded);

  // Decode the URI-encoded key and convert it to binary data
  const washedKey = new Uint8Array(atob(decodeURIComponent(publicKeyUriEncoded)).split('').map(c => c.charCodeAt(0)));

  // Get the original binary key data (assuming it's stored in a variable named 'originalKey')
  const originalKey = new Uint8Array(publicKey);

  // Compare the two keys
  const transmissionTest = JSON.stringify(washedKey) === JSON.stringify(originalKey);
  console.log('Transmission encoding/decoding test status: ', transmissionTest);
} catch (error) {
  console.error('Transmission encoding/decoding test failed: ', error);
}
    // Test encryption/decryption with the server keys
    const { publicKey: serverPublicKey, privateKey: serverPrivateKey } = await ServerCommunication.getServerKeys(hashedPassword);

    // Encrypt the message with the server public key
    const encryptedWithServerKey = await window.crypto.subtle.encrypt({ name: 'RSA-OAEP' }, serverPublicKey, message);

    // Decrypt the message with the local private key
    const decryptedWithServerKey = Uint8Array(await window.crypto.subtle.decrypt({ name: 'RSA-OAEP' }, serverPrivateKey, encryptedWithServerKey));

    // Compare the original message and the decrypted message to ensure they are the same
    let isEqualWithServerKey = true;
    for (let i = 0; i < message.length; i++) {
      if (message[i] !== decryptedWithServerKey[i]) {
        isEqualWithServerKey = false;
        break;
      }
    }

    console.log('Server encryption/decryption test passed:', isEqualWithServerKey);
  } catch (error) {
    console.error('Encryption/decryption test failed:', error);
  }
}
static async keyToPEM(key, type) {
 const exported = type === 'private'
    ? crypto.subtle.exportKey('pkcs8', key)
    : crypto.subtle.exportKey('spki', key);
  return exported.then((raw) => {
    const base64 = btoa(String.fromCharCode(...new Uint8Array(raw)));
    let pem = '';
    let i = 0;
    while (i < base64.length) {
      pem += `${base64.slice(i, i += 64)}\n`;
    }
    return `-----BEGIN ${type.toUpperCase()} KEY-----\n${pem}-----END ${type.toUpperCase()} KEY-----\n`;
  });
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
    const gcmKey = await ServerCommunication.deriveGcmKey(previousTag, randomIv);

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
static async decryptWithGCM(encryptedData, gcmConfig) {
  const iv = gcmConfig.iv;
  const tag = gcmConfig.tag;
  const algorithm = { name: 'AES-GCM', iv: iv, tagLength: 128, tag: tag };
  const key = await window.crypto.subtle.importKey('raw', iv, {name: 'AES-GCM'}, false, ['encrypt', 'decrypt']);
  const decrypted = await window.crypto.subtle.decrypt(algorithm, key, encryptedData);
  return new Uint8Array(decrypted);
}
static async initializeAES(password){
try {
		let salt, key, hashedPassword, iv;
		hashedPassword = CryptoJS.SHA256(password).toString();
		salt = CryptoJS.lib.WordArray.random(16);
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
static async pubPrivatePayload(password, payload) {
	
	
  // Fetch the server's public key
  let encryptedCombinedData;
 

 let keyPair = await ServerCommunication.makeRSAKeys();
  
  const {salt, key, hashedPassword, iv} = await ServerCommunication.initializeAES("mysecretkey");
  
  await ServerCommunication.testEncryption(hashedPassword);
  return null;
  const {publicKey: serverPublicKey, chunkSize} = await ServerCommunication.getServerKey(hashedPassword);
try{
	let encryptedPayload, encryptedPassword, clientPublicKey, encryptedClientPublicKey, combinedData;

try {
  // Encrypt the Payload string using AES-256-CBC
  encryptedPayload = CryptoJS.AES.encrypt(payload, key, { iv: iv }).ciphertext;
} catch (error) {
  console.error(error);
}

try {
  // Encrypt the hashed password using AES-256-CBC
  encryptedPassword = CryptoJS.AES.encrypt(hashedPassword, key, { iv: iv }).ciphertext;
} catch (error) {
  console.error(error);
}

try {
  // Encrypt the client's public key using AES-256-CBC
  clientPublicKey = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
} catch (error) {
  console.log(error);
}

try {
  encryptedClientPublicKey = CryptoJS.AES.encrypt( new TextDecoder().decode(clientPublicKey), key, { iv: iv }).ciphertext;
} catch (error) {
  console.error(error);
}
console.warn(encryptedPayload, encryptedPassword, encryptedClientPublicKey, iv, salt, key);
let encPass;
try {
  // Pack the IV, encrypted hashed password, encrypted client public key, and encrypted payload into a single WordArray
const encSalt = CryptoJS.enc.Base64.parse(CryptoJS.enc.Hex.parse(salt.toString()).toString(CryptoJS.enc.Base64));
console.log(encSalt);
const encIv = CryptoJS.enc.Base64.parse(CryptoJS.enc.Hex.parse(iv.toString()).toString(CryptoJS.enc.Base64));
console.log(encIv);
  encPass = CryptoJS.enc.Base64.parse(CryptoJS.enc.Hex.parse(encryptedPassword.toString()).toString(CryptoJS.enc.Base64));
  const encPubKey = CryptoJS.enc.Base64.parse(CryptoJS.enc.Hex.parse(encryptedClientPublicKey.toString()).toString(CryptoJS.enc.Base64));
  const encPayload = CryptoJS.enc.Base64.parse(CryptoJS.enc.Hex.parse(encryptedPayload.toString()).toString(CryptoJS.enc.Base64));
  //const dataIndices =  
  combinedData = CryptoJS.lib.WordArray.create()
    .concat(encSalt)
    .concat(encIv)
    .concat(encPass)
    .concat(encPubKey)
    .concat(encPayload);
} catch (error) {
  console.error("error concatting payload", error);
}
console.log('salt: ' + salt.toString(CryptoJS.enc.Hex));
console.log('iv: ' + iv.toString(CryptoJS.enc.Hex));
console.log('key: ' + key.toString(CryptoJS.enc.Hex));
console.log('encryptedPassword: ' + encryptedPassword.toString(CryptoJS.enc.Hex));
console.log('encryptedClientPublicKey: ' + encryptedClientPublicKey.toString(CryptoJS.enc.Base64));
console.log('encryptedPayload: ' + encryptedPayload.toString(CryptoJS.enc.Base64));
console.warn(combinedData);


try {
 const combinedDataBase64 = CryptoJS.enc.Base64.stringify(combinedData);

const chunkedData = combinedDataBase64.match(new RegExp('.{1,' + chunkSize + '}', 'g'));
console.error(chunkedData);
const encryptedChunks = [];

for (let i = 0; i < chunkedData.length; i++) {
	console.warn("1");
  const chunk = chunkedData[i];


 const encryptedChunk = await window.crypto.subtle.encrypt({
    name: "RSA-OAEP",
      hash: "SHA-256"
  }, serverPublicKey, new TextEncoder().encode(chunk));
  encryptedChunks.push(btoa(String.fromCharCode(...new Uint8Array(encryptedChunk))));
}
console.error(encryptedChunks);
encryptedCombinedData = JSON.stringify(encryptedChunks);
return encryptedCombinedData;

} catch (error) {
  console.error("error returning", error);
  return null;
}
  // Encode the combined data using Base64
  //return combinedData;//CryptoJS.enc.Base64.stringify(combinedData);
}catch(error){
	console.error("Error encoding at line ", line, ":",error);
}

try {
  // Decrypt the concatenated data
  const encryptedChunks = JSON.parse(encryptedCombinedData);
  const decryptedChunks = [];

  for (let i = 0; i < encryptedChunks.length; i++) {
    const encryptedChunk = atob(encryptedChunks[i]);
    const decryptedChunk = await window.crypto.subtle.decrypt({
      name: "RSA-OAEP",
      hash: "SHA-256"
    }, keyPair.privateKey, new Uint8Array(Array.from(encryptedChunk)).buffer);
    decryptedChunks.push(new TextDecoder().decode(decryptedChunk));
  }

  const decryptedCombinedData = CryptoJS.enc.Base64.parse(decryptedChunks.join(""));

  // Extract the salt, IV, encrypted hashed password, encrypted client public key, and encrypted payload from the decrypted data
  const saltSize = 16;
  const ivSize = 16;
  const encryptedPasswordSize = 32;
  const encryptedClientPublicKeySize = 294;
  const salt = decryptedCombinedData.words.slice(0, saltSize / 4);
  const iv = decryptedCombinedData.words.slice(saltSize / 4, (saltSize + ivSize) / 4);
  const encryptedPassword = decryptedCombinedData.words.slice((saltSize + ivSize) / 4, (saltSize + ivSize + encryptedPasswordSize) / 4);
  const encryptedClientPublicKey = decryptedCombinedData.words.slice((saltSize + ivSize + encryptedPasswordSize) / 4, (saltSize + ivSize + encryptedPasswordSize + encryptedClientPublicKeySize) / 4);
  const encryptedPayload = decryptedCombinedData.words.slice((saltSize + ivSize + encryptedPasswordSize + encryptedClientPublicKeySize) / 4);

  // Decrypt the hashed password, client public key, and payload using the key derived from the password and salt, and the IV
  const key = CryptoJS.PBKDF2(hashedPassword, CryptoJS.enc.Hex.parse(salt.toString(CryptoJS.enc.Hex)), {
    keySize: 256/32,
    iterations: iterations
  });

  const password = CryptoJS.AES.decrypt({
    ciphertext: CryptoJS.lib.WordArray.create(encryptedPassword)
  }, key, {
    iv: CryptoJS.lib.WordArray.create(iv)
  }).toString(CryptoJS.enc.Utf8);

  const clientPublicKey = await window.crypto.subtle.importKey("spki", new TextEncoder().encode(CryptoJS.AES.decrypt({
    ciphertext: CryptoJS.lib.WordArray.create(encryptedClientPublicKey)
  }, key, {
    iv: CryptoJS.lib.WordArray.create(iv)
  }).toString(CryptoJS.enc.Utf8)), {
    name: "RSA-OAEP",
    hash: "SHA-256"
  }, false, ["encrypt"]);

  const payload = CryptoJS.AES.decrypt({
    ciphertext: CryptoJS.lib.WordArray.create(encryptedPayload)
  }, key, {
    iv: CryptoJS.lib.WordArray.create(iv)
  }).toString(CryptoJS.enc.Utf8);

  // Compare the original password and payload with the decrypted values
  console.log("Original password:", password);
  console.log("Original payload:", payload);
  
  return encryptedCombinedData;
} catch (error) {
  console.error("error decrypting payload", error);
}



}

}