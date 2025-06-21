<?php

include_once("key.php");
function generateGcmCrc($sessionID, $encryptedData, $encryptedConfig) {
    // Decrypt the configuration data
    
    $keyDir = __DIR__ . "/keys";
    $publicKeyFile = $keyDir . "/" . $sessionID . "_public.pem";
    
    error_log("This is our encoded secret data in base64: " . $encryptedConfig);
    $decryptedConfig = '';
    
    $encryptedConfig = base64_decode($encryptedConfig);
    //error_log("This is our encoded secret data in binary: " . $encryptedConfig);
    $paddingStyles = [
        OPENSSL_PKCS1_PADDING,
        OPENSSL_SSLV23_PADDING,
        OPENSSL_PKCS1_OAEP_PADDING,
        OPENSSL_NO_PADDING
    ];
    
    // Retrieve private key
    $privateKey = getPrivateKey($sessionID);
    
    // Retrieve public key
    $publicKey = openssl_get_publickey(file_get_contents($publicKeyFile));
    
    // Encryption/decryption test
    $data = "test data";
    $encryptedData = '';
    $result = openssl_public_encrypt($data, $encryptedData, $publicKey);
    if (!$result) {
        die("Failed to encrypt test data");
    }
    
    $decryptedData = '';
    $result = openssl_private_decrypt($encryptedData, $decryptedData, $privateKey);
    if (!$result) {
        die("Failed to decrypt test data");
    }
    
    if ($decryptedData !== $data) {
        die("Encryption/decryption test failed");
    }
    
    // Decrypt the configuration data
    $decryptedConfig = '';

    $errorCount = 0;
    
    foreach ($paddingStyles as $padding) {
        $decryptedConfig = '';
        $result = openssl_private_decrypt($encryptedConfig, $decryptedConfig, $privateKey, $padding);
        if ($result) {
            error_log("Decryption succeeded with padding $padding");
            break;
        }
        error_log("Decryption failed with padding $padding");
        $errorCount++;
    }
    
    if ($errorCount === count($paddingStyles)) {
        die("Failed to decrypt configuration data with any padding style");
    } else {
        error_log("This is the decrypted data in base64: " . base64_encode($decryptedConfig));
    }
    $encodedConfig = utf8_encode($decryptedConfig);
    error_log($encodedConfig);
    $config = json_decode($encodedConfig);
    if (!$config) {
        die("Failed to decode configuration data");
    }
    
    // Derive the GCM key
    $hashedPassword = $config->hashedPassword;
    $salt = $config->salt;
    $key = deriveKey($hashedPassword, $salt);
    
    // Decrypt the encrypted data and verify the CRC
    $data = openssl_decrypt(base64_decode($encryptedData), "aes-256-gcm", $key, OPENSSL_RAW_DATA, $config->iv, $config->tag, $config->tagLength);
    $crc = hash("sha256", $data);
    
    // Encrypt the CRC using the same key as the data
    $encryptedCrc = openssl_encrypt($crc, "aes-256-gcm", $key, OPENSSL_RAW_DATA, $config->iv, $tag);
    
    // Append the tag to the encrypted CRC
    $encryptedCrc .= $tag;
    
    return $encryptedCrc;
}

function getGcmCrc() {
    $sessionID = $_GET["sessionID"];
    $encryptedData = $_GET["encryptedData"];
    $encryptedConfig = $_GET["encryptedConfig"];
    
    // Generate the GCM key and calculate the CRC
    $encryptedCrc = generateGcmCrc($sessionID, $encryptedData, $encryptedConfig);
    
    return $encryptedCrc;
}

// Get the GCM CRC string
$encryptedCrc = getGcmCrc();
error_log("the return is: " . $encryptedCrc);

// Return the CRC to the client
header("Content-Type: text/plain");

echo base64_encode($encryptedCrc);
?>