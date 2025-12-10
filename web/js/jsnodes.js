import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "AudioSeparationNodes.AudioNewList.DynamicInputs",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== "AudioNewList") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      this._type = "AUDIO";
      this.inputs_offset = 0;
      this.addWidget("button", "Update inputs", null, () => {
        if (!this.inputs) this.inputs = [];

        const widget = this.widgets?.find((w) => w.name === "inputcount");
        const target_number_of_inputs = widget ? widget.value : 2;
        const num_inputs = this.inputs.filter((input) => input.type === this._type).length;

        if (target_number_of_inputs === num_inputs) return; // already set

        if (target_number_of_inputs < num_inputs) {
          // Remove only trailing AUDIO inputs beyond the target
          for (let i = this.inputs.length - 1; i >= 0; i--) {
            if (this.inputs.filter((input) => input.type === this._type).length <= target_number_of_inputs) break;
            if (this.inputs[i].type === this._type) {
              this.removeInput(i);
            }
          }
        } else {
          for (let i = num_inputs + 1; i <= target_number_of_inputs; ++i) {
            this.addInput(`audio_${i}`, this._type);
          }
        }
      });

      return r;
    };
  },
});
