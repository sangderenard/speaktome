class Message {
  constructor(data) {
    if (data instanceof ArrayBuffer) {
      this.dataView = new DataView(data);
      this.offset = 0;
    } else if (data instanceof DataView) {
      this.dataView = data;
      this.offset = data.byteOffset;
    } else {
      throw new Error('Invalid data type for Message constructor.');
    }

    this.recordHeader = this.dataView.getUint8(this.offset);
    this.isDefinition = (this.recordHeader & 0xF0) === 0x40;
    this.isCompressed = !this.isDefinition && ((this.recordHeader & 0x80) >> 7) === 1;
    this.localMsgType = this.recordHeader & 0x0F;
    this.devDataFlag = (this.recordHeader >> 4) & 0x01;
    this.devData = null;
    this.recordContent = null;
  }
}