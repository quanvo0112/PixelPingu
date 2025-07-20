class PixelArtCanvas {
	constructor() {
		this.canvas = document.getElementById("artCanvas");
		this.ctx = this.canvas.getContext("2d");
		this.previewCanvas = document.getElementById("previewCanvas");
		this.previewCtx = this.previewCanvas.getContext("2d");

		this.gridSize = 128;
		this.pixelSize = this.canvas.width / this.gridSize; 

		this.pixelData = new Uint8ClampedArray(
			this.gridSize * this.gridSize * 4,
		);
		for (let i = 0; i < this.pixelData.length; i += 4) {
			this.pixelData[i] = 255;
			this.pixelData[i + 1] = 255;
			this.pixelData[i + 2] = 255;
			this.pixelData[i + 3] = 255;
		}

		this.currentColor = "#000000";
		this.brushSize = 1;
		this.isDrawing = false;
		this.isEraser = false;

		this.setupEventListeners();
		this.setupControls();
		this.render();
		this.updatePreview();
	}

	setupEventListeners() {
		this.canvas.addEventListener("mousedown", (e) => this.startDrawing(e));
		this.canvas.addEventListener("mousemove", (e) => this.draw(e));
		this.canvas.addEventListener("mouseup", () => this.stopDrawing());
		this.canvas.addEventListener("mouseleave", () => this.stopDrawing());

		this.canvas.addEventListener("touchstart", (e) => {
			e.preventDefault();
			const touch = e.touches[0];
			const mouseEvent = new MouseEvent("mousedown", {
				clientX: touch.clientX,
				clientY: touch.clientY,
			});
			this.canvas.dispatchEvent(mouseEvent);
		});

		this.canvas.addEventListener("touchmove", (e) => {
			e.preventDefault();
			const touch = e.touches[0];
			const mouseEvent = new MouseEvent("mousemove", {
				clientX: touch.clientX,
				clientY: touch.clientY,
			});
			this.canvas.dispatchEvent(mouseEvent);
		});

		this.canvas.addEventListener("touchend", (e) => {
			e.preventDefault();
			const mouseEvent = new MouseEvent("mouseup", {});
			this.canvas.dispatchEvent(mouseEvent);
		});
	}

	setupControls() {
		const colorPicker = document.getElementById("colorPicker");
		colorPicker.addEventListener("change", (e) => {
			this.currentColor = e.target.value;
			this.isEraser = false;
			document.getElementById("eraser").classList.remove("active");
		});

		const brushSize = document.getElementById("brushSize");
		brushSize.addEventListener("change", (e) => {
			this.brushSize = parseInt(e.target.value);
		});

		document.getElementById("clearCanvas").addEventListener("click", () => {
			this.clearCanvas();
		});

		document.getElementById("fillCanvas").addEventListener("click", () => {
			this.fillCanvas();
		});

		document.getElementById("eraser").addEventListener("click", (e) => {
			this.isEraser = !this.isEraser;
			e.target.classList.toggle("active");
		});

		document
			.getElementById("submitForm")
			.addEventListener("submit", (e) => {
				e.preventDefault();
				this.submitArtwork();
			});
	}

	getGridPosition(e) {
		const rect = this.canvas.getBoundingClientRect();
		const x = Math.floor((e.clientX - rect.left) / this.pixelSize);
		const y = Math.floor((e.clientY - rect.top) / this.pixelSize);
		return {
			x: Math.max(0, Math.min(x, this.gridSize - 1)),
			y: Math.max(0, Math.min(y, this.gridSize - 1)),
		};
	}

	startDrawing(e) {
		this.isDrawing = true;
		this.draw(e);
	}

	draw(e) {
		if (!this.isDrawing) return;

		const pos = this.getGridPosition(e);
		const color = this.isEraser
			? [255, 255, 255, 255]
			: this.hexToRgba(this.currentColor);

		for (let dx = 0; dx < this.brushSize; dx++) {
			for (let dy = 0; dy < this.brushSize; dy++) {
				const x = pos.x + dx;
				const y = pos.y + dy;

				if (
					x >= 0 &&
					x < this.gridSize &&
					y >= 0 &&
					y < this.gridSize
				) {
					this.setPixel(x, y, color);
				}
			}
		}

		this.render();
		this.updatePreview();
	}

	stopDrawing() {
		this.isDrawing = false;
	}

	setPixel(x, y, color) {
		const index = (y * this.gridSize + x) * 4;
		this.pixelData[index] = color[0];
		this.pixelData[index + 1] = color[1];
		this.pixelData[index + 2] = color[2];
		this.pixelData[index + 3] = color[3];
	}

	hexToRgba(hex) {
		const r = parseInt(hex.substr(1, 2), 16);
		const g = parseInt(hex.substr(3, 2), 16);
		const b = parseInt(hex.substr(5, 2), 16);
		return [r, g, b, 255];
	}

	render() {
		this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

		this.ctx.imageSmoothingEnabled = false;

		const tempCanvas = document.createElement("canvas");
		tempCanvas.width = this.gridSize;
		tempCanvas.height = this.gridSize;
		const tempCtx = tempCanvas.getContext("2d");

		const imageData = tempCtx.createImageData(this.gridSize, this.gridSize);
		imageData.data.set(this.pixelData);
		tempCtx.putImageData(imageData, 0, 0);

		this.ctx.drawImage(
			tempCanvas,
			0,
			0,
			this.gridSize,
			this.gridSize,
			0,
			0,
			this.canvas.width,
			this.canvas.height,
		);

		this.drawGrid();
	}

	drawGrid() {
		this.ctx.strokeStyle = "rgba(0, 0, 0, 0.1)";
		this.ctx.lineWidth = 1;

		for (let x = 0; x <= this.gridSize; x++) {
			const xPos = x * this.pixelSize;
			this.ctx.beginPath();
			this.ctx.moveTo(xPos, 0);
			this.ctx.lineTo(xPos, this.canvas.height);
			this.ctx.stroke();
		}

		for (let y = 0; y <= this.gridSize; y++) {
			const yPos = y * this.pixelSize;
			this.ctx.beginPath();
			this.ctx.moveTo(0, yPos);
			this.ctx.lineTo(this.canvas.width, yPos);
			this.ctx.stroke();
		}
	}

	updatePreview() {
		this.previewCtx.clearRect(
			0,
			0,
			this.previewCanvas.width,
			this.previewCanvas.height,
		);
		this.previewCtx.imageSmoothingEnabled = false;

		const tempCanvas = document.createElement("canvas");
		tempCanvas.width = this.gridSize;
		tempCanvas.height = this.gridSize;
		const tempCtx = tempCanvas.getContext("2d");

		const imageData = tempCtx.createImageData(this.gridSize, this.gridSize);
		imageData.data.set(this.pixelData);
		tempCtx.putImageData(imageData, 0, 0);

		this.previewCtx.drawImage(
			tempCanvas,
			0,
			0,
			this.gridSize,
			this.gridSize,
			0,
			0,
			this.previewCanvas.width,
			this.previewCanvas.height,
		);
	}

	clearCanvas() {
		for (let i = 0; i < this.pixelData.length; i += 4) {
			this.pixelData[i] = 255;
			this.pixelData[i + 1] = 255;
			this.pixelData[i + 2] = 255;
			this.pixelData[i + 3] = 255;
		}
		this.render();
		this.updatePreview();
	}

	fillCanvas() {
		const color = this.hexToRgba(this.currentColor);
		for (let i = 0; i < this.pixelData.length; i += 4) {
			this.pixelData[i] = color[0];
			this.pixelData[i + 1] = color[1];
			this.pixelData[i + 2] = color[2];
			this.pixelData[i + 3] = color[3];
		}
		this.render();
		this.updatePreview();
	}

	submitArtwork() {
		const canvasData = Array.from(this.pixelData);

		const submitBtn = document.querySelector(
			'#submitForm button[type="submit"]',
		);
		const originalText = submitBtn.textContent;
		submitBtn.disabled = true;
		submitBtn.innerHTML = '<span class="loading"></span> Submitting...';

		fetch("/submit_artwork", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				canvas_data: canvasData,
			}),
		})
			.then((response) => response.json())
			.then((data) => {
				if (data.success) {
					const resultDiv = document.getElementById("submissionResult");
					if (resultDiv) {
						resultDiv.innerHTML = "";
						let html = `
							<div class='result-message card shadow-sm p-3 mb-2 bg-light rounded text-center'>
								<div class='ai-score fw-bold text-primary' style='font-size:1.2em;'><i class='fas fa-paint-brush'></i> Artwork Score: <span class='fw-bold'>${data.judge_score !== undefined ? data.judge_score : ''}</span></div>
								${data.flag_part ? `<div class='flag-part mt-3 alert alert-warning fw-bold' style='font-size:1.2em;'><i class='fas fa-trophy'></i> Flag part: <span class='fw-bold'>${data.flag_part}</span></div>` : ''}
							</div>
						`;
						resultDiv.innerHTML = html;
					}
				} else {
					alert("Error submitting artwork: " + data.error);
				}
			})
			.catch((error) => {
				console.error("Error:", error);
				alert("Error submitting artwork. Please try again.");
			})
			.finally(() => {
				submitBtn.disabled = false;
				submitBtn.textContent = originalText;
			});
	}
}

document.addEventListener("DOMContentLoaded", function () {
	if (document.getElementById("artCanvas")) {
		new PixelArtCanvas();
	}
});
