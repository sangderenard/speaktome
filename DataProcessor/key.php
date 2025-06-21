<?php

ini_set('display_errors', 1);
error_reporting(E_ALL);
function generateCertificate($keypair) {
    // Set up the DN (Distinguished Name) for the certificate
    $dn = array(
        "countryName" => "US",
        "stateOrProvinceName" => "California",
        "localityName" => "San Francisco",
        "organizationName" => "Example Inc.",
        "organizationalUnitName" => "IT Department",
        "commonName" => "example.com",
        "emailAddress" => "admin@example.com"
    );
    
    // Generate the X.509 certificate
    $cert = openssl_csr_new($dn, $keypair);
    $cert = openssl_csr_sign($cert, null, $keypair, 365);
    openssl_x509_export($cert, $certout);
    
    // Return the certificate
    return $certout;
}
function hashPasscode($hash, $passcode) {
    $hashFunctions = [
        function ($input) {
            return hash('sha1', $input);
        },
        function ($input) {
            return hash('sha256', $input);
        },
        function ($input) {
            return hash('sha384', $input);
        },
        function ($input) {
            return hash('sha512', $input);
        }
        ];
    
    $hashBits = '';
    foreach (str_split($hash) as $byte) {
        $hashBits .= str_pad(decbin(ord($byte)), 8, '0', STR_PAD_LEFT);
    }
    
    $hashFunctionIndices = [];
    for ($i = 0; $i < strlen($hashBits); $i += 2) {
        $bitPair = substr($hashBits, $i, 2);
        $hashFunctionIndices[] = bindec($bitPair);
    }
    
    $hashedPasscode = $passcode;
    foreach ($hashFunctionIndices as $index) {
        $hashFunction = $hashFunctions[$index];
        $hashedPasscode = $hashFunction($hashedPasscode);
    }
    return $passcode;
    return $hashedPasscode;
}
function generateKeyPair($sessionID) {
  //  testEncryptDecrypt();
    $overwrite = 1;
    $keyDir = __DIR__ . "/keys";
    $publicKeyFile = $keyDir . "/" . $sessionID . "_public.pem";
    $clientPublicKeyFile = $keyDir . "/" . $sessionID . "_client_public.pem";
    $privateKeyFile = $keyDir . "/" . $sessionID . "_private.p12";
    $clientKeyFile = $keyDir . "/" . $sessionID . "_client.p12";
    $passcodeFile = $keyDir . "/" . $sessionID . ".key";
    $clientPasscodeFile = $keyDir . "/" . $sessionID . "_client.key";
    $publicKey = "";
    $passphrase = "passcode";
    if (!$overwrite && file_exists($publicKeyFile) && file_exists($passcodeFile)) {
        // Read the public key from the file
        $publicKey = file_get_contents($publicKeyFile);
    } else {
        // Generate a new RSA key pair for the server
        $config = array(
            "digest_alg" => "sha512",
            "private_key_bits" => 4096,
            "private_key_type" => OPENSSL_KEYTYPE_RSA,
        );
        $keypair = openssl_pkey_new($config);
        $clientKeypair = openssl_pkey_new($config);
        $certificate = generateCertificate($keypair);
        $clientCertificate = generateCertificate($clientKeypair);
        
        // Get the private key from the keypair
        $privateKey = '';
        $clientPrivateKey = '';
        openssl_pkey_export($keypair, $privateKey);
        openssl_pkey_export($clientKeypair, $clientPrivateKey);
        $privateKeyResource = openssl_pkey_get_private($privateKey);
        $clientPrivateKeyResource = openssl_pkey_get_private($clientPrivateKey);
        // Get the private key and save it to a PKCS12 file
        openssl_pkcs12_export($certificate, $pfx, $privateKeyResource, $passphrase);
        openssl_pkcs12_export($clientCertificate, $cpfx, $clientPrivateKeyResource, $passphrase);
        error_log($pfx);
        
        // Encrypt and mask the PKCS12 file
        [$mask , $encryptedPfx ] = encryptToSave($pfx, $sessionID);
        [$clientMask, $encryptedClientPfx ] = encryptToSave($cpfx, $sessionID);
        
        file_put_contents($privateKeyFile, $encryptedPfx);
        file_put_contents($passcodeFile, $mask);
        
        // Get the public key and save it to a file
        $publicKey = openssl_pkey_get_details($keypair)["key"];
        $clientPublicKey = openssl_pkey_get_details($clientKeypair)["key"];
        file_put_contents($publicKeyFile, $publicKey);
        file_put_contents($clientPublicKeyFile, $clientPublicKey);
        
        openssl_pkey_export($clientKeypair, $clientPrivateKeyPem);
    }
    
    
    return [ $publicKey, $clientPublicKey, $clientPrivateKeyPem ];
}
function deriveKeyAndIV($input, $salt, $hash = 'sha256') {
    $keySize = 32; // 32 bytes = 256 bits
    $ivSize = 16; // 16 bytes = 128 bits
    $iterations = 1; // number of iterations
    
    $keyAndIV = hash_pbkdf2($hash, $input, $salt, $iterations, $keySize + $ivSize, true);
    
    $key = substr($keyAndIV, 0, $keySize);
    $iv = substr($keyAndIV, $keySize, $ivSize);
    $key = base64_decode('7xv2Cq3V8H50zQhOED7wv/gO/umG8dL42iCHrN52NvI=');
    $iv = base64_decode('IaU6z4UwPsdU6DZUdHYH2Q==');
    return array('key' => $key, 'iv' => $iv);
}
function openssl_shell_encrypt($data, $method, $key, $options = 0, $iv = null, &$tag = null, $tag_length = 16) {
    $iv_param = $iv !== null ? "-iv $iv" : '';
    $tag_param = $tag !== null ? "-tag $tag" : '';
    $cmd = "echo \"$data\" | openssl enc -$method -K $key $iv_param $tag_param -base64";
    exec($cmd, $output);
    $result = implode("\n", $output);
    if ($tag !== null) {
        list($encryptedData, $tag) = explode(':', $result);
        $tag = base64_decode($tag);
        return base64_decode($encryptedData);
    } else {
        return base64_decode($result);
    }
}

function openssl_shell_decrypt($data, $method, $key, $options = 0, $iv = null, $tag = null, $tag_length = 16) {
    $iv_param = $iv !== null ? "-iv $iv" : '';
    $tag_param = $tag !== null ? "-tag $tag" : '';
    $cmd = "echo \"$data\" | openssl enc -$method -K $key $iv_param $tag_param -d -base64";
    exec($cmd, $output, $retval);
    if ($retval !== 0) {
        return false;
    }
    return implode("\n", $output);
}
function simpleEncrypt($data, $salt, $passcode, $cipher = 'chacha20', $hash = 'sha256'){
    $derived = deriveKeyAndIV($passcode, $salt, $hash);
    error_log("Key and IV: " . base64_encode($derived['key']) . ":" . base64_encode($derived['iv']));
    switch ($cipher) {
        case 'aes-256-cbc':
            $options = 0;
            break;
        case 'aes-256-ctr':
            $options = OPENSSL_RAW_DATA;
            break;
        case 'aes-256-gcm':
            $options = OPENSSL_RAW_DATA;
            $tag_length = 16;
            break;
        case 'camellia-256-cbc':
            $options = 0;
            break;
        case 'chacha20':
            $options = OPENSSL_RAW_DATA;
            break;
        default:
            throw new Exception('Invalid cipher');
    }
    
    if ($cipher === 'aes-256-gcm') {
        $encryptedData = openssl_encrypt($data, $cipher, $derived['key'], $options, $derived['iv'], $tag, $tag_length);
        
        return base64_encode($encryptedData) . ':' . base64_encode($tag);
    } else {
        $tag = random_bytes(16);
        $encryptedData =  openssl_encrypt($data, $cipher, $derived['key'], $options, $derived['iv']);
        
        return base64_encode($encryptedData) . ':' . base64_encode($tag);
    }
}

function simpleDecrypt($data, $salt, $passcode, $cipher = 'chacha20', $hash = 'sha256'){
    $derived = deriveKeyAndIV($passcode, $salt, $hash);
    
    switch ($cipher) {
        case 'aes-256-cbc':
            $options = 0;
            break;
        case 'aes-256-ctr':
            $options = OPENSSL_RAW_DATA;
            break;
        case 'aes-256-gcm':
            $options = OPENSSL_RAW_DATA;
            $tag_length = 16;
            break;
        case 'camellia-256-cbc':
            $options = 0;
            break;
        case 'chacha20':
            $options = OPENSSL_RAW_DATA;
            break;
        default:
            throw new Exception('Invalid cipher');
    }
    
    $data_parts = explode(':', $data);
    $encryptedData = base64_decode($data_parts[0]);
    $tag = base64_decode($data_parts[1]);
    
    
    if ($cipher === 'aes-256-gcm') {
        return openssl_decrypt($encryptedData, $cipher, $derived['key'], $options, $derived['iv'], $tag, $tag_length);
    } else {
        return openssl_decrypt($encryptedData, $cipher, $derived['key'], $options, $derived['iv']);
    }
}
function testEncryptDecrypt() {
    $hashAlgorithms = array('sha256', 'sha384', 'sha512');
    $ciphers = array('aes-256-cbc', 'aes-256-ctr', 'aes-256-gcm', 'camellia-256-cbc', 'chacha20');
    
    $passcode = random_bytes(16);
    //$data = random_bytes(1024);
    $data = '';
    for ($i = 0; $i < 756; $i++) {
        $data .= chr($i % 256);
    }
    $salt = random_bytes(16);
    
    foreach ($hashAlgorithms as $hashAlgorithm) {
        foreach ($ciphers as $cipher) {
            //$derived = deriveKeyAndIV($passcode, $salt, $hashAlgorithm);
            
            $encrypted = simpleEncrypt($data, $salt, $passcode, $cipher, $hashAlgorithm);
            error_log("Encrypted Test Data: " . $encrypted);
            $decrypted = simpleDecrypt($encrypted, $salt, $passcode, $cipher, $hashAlgorithm);
            
            if ($decrypted === $data) {
                error_log( "PASS: $cipher with $hashAlgorithm\n");
            } else {
                error_log( "FAIL: $cipher with $hashAlgorithm\n");
            }
        }
    }
}
function encryptToSave($content, $passcode){
    $contentSize = strlen($content);
    $mask = openssl_random_pseudo_bytes($contentSize);
    
    $encryptedContents = simpleEncrypt($content, substr($mask, 0, 16), $passcode);
    $encryptedContents = $mask ^ $encryptedContents;
    
    return [ $encryptedContents, $mask ];
}
function decryptFromSave($file, $mask, $passcode){
    $file = $file ^ $mask;
    return simpleDecrypt($file, substr($mask, 0, 16), $passcode);
    
}

function getClientPrivateKey($sessionID){
    $passphrase = "passcode";
    $keyDir = __DIR__ . "/keys";
    $clientPrivateKeyFile = $keyDir . "/" . $sessionID . "_client.p12";
    $passcodeFile = $keyDir . "/" . $sessionID . "_client.key";
    
    $mask = file_get_contents($passcodeFile);
    $encryptedPfx = file_get_contents($clientPrivateKeyFile);
    
    $pfx = decryptFromSave($encryptedPfx, $mask, $sessionID);
    $certs = array();
    if (openssl_pkcs12_read($pfx, $certs, $passphrase)) {
        $privateKey = $certs["pkey"];
    } else{
        throw new Exception("Unable to load private key from PKCS12 file.");
    }
}

function getPrivateKey($sessionID){
    $passphrase = "passcode";
    $keyDir = __DIR__ . "/keys";
    $privateKeyFile = $keyDir . "/" . $sessionID . "_private.p12";
    $passcodeFile = $keyDir . "/" . $sessionID . ".key";
    
    // Decrypt the encrypted PKCS12 file using the private key
    $mask = file_get_contents($passcodeFile);
    $encryptedPfx = file_get_contents($privateKeyFile);
    
    $pfx = decryptFromSave($encryptedPfx, $mask, $sessionID);
    error_log($pfx);
    // Load the private key from the PKCS12 file
    $certs = array();
    if (openssl_pkcs12_read($pfx, $certs, $passphrase)) {
        $privateKey = $certs["pkey"];
    } else {
        throw new Exception("Unable to load private key from PKCS12 file.");
    }
    
    return $privateKey;
}
function decryptPEM($input, $passphrase) {
    // Convert PEM string to OpenSSL key
    $key = openssl_pkey_get_private($input, $passphrase);
    
    // Export the key as an unencrypted PEM string
    $decryptedKey = '';
    openssl_pkey_export($key, $decryptedKey);
    
    // Cleanup
    openssl_free_key($key);
    
    return $decryptedKey;
}


?>