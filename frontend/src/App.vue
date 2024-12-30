<template>
  <div class="bg-black h-screen w-screen overflow-hidden">
    <div class="relative isolate px-6 lg:px-8 h-full">
      <div
        class="absolute inset-x-0 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-50"
        aria-hidden="true"
      >
        <div
          class="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%-10rem)] sm:w-[72.1875rem]"
          style="
            clip-path: polygon(
              74.1% 44.1%,
              100% 61.6%,
              97.5% 26.9%,
              85.5% 0.1%,
              80.7% 2%,
              72.5% 32.5%,
              60.2% 62.4%,
              52.4% 68.1%,
              47.5% 58.3%,
              45.2% 34.5%,
              27.5% 76.7%,
              0.1% 64.9%,
              17.9% 100%,
              27.6% 76.8%,
              76.1% 97.7%,
              74.1% 44.1%
            );
          "
        />
      </div>
      <div v-if="!this.isEdit" class="mx-auto max-w-2xl py-24 sm:py-48 lg:py-56 h-full">
        <div class="hidden sm:mb-8 sm:flex sm:justify-center">
          <div
            class="relative rounded-full px-3 py-1 text-sm/6 text-gray-600 ring-1 ring-white/10 hover:ring-white/20"
          >
            UCAS Advanced OS Final Project 2024 | 舒恒 · 郭晓燕 · 刘梓铭
          </div>
        </div>
        <div class="text-center">
          <h1
            class="flex items-center justify-center text-balance text-9xl tracking-tight text-yellow-300 sm:text-7xl font-extrabold"
          >
            <svg
              viewBox="0 0 24 24"
              width="100"
              height="100"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              stroke="#ffcd42"
              transform="matrix(-1, 0, 0, 1, 0, 0)"
            >
              <path
                d="M3.84453 3.84453C2.71849 4.97056 2.71849 6.79623 3.84453 7.92226L5.43227 9.51C5.44419 9.49622 5.45669 9.48276 5.46978 9.46967L9.46978 5.46967C9.48284 5.45662 9.49625 5.44415 9.50999 5.43226L7.92226 3.84453C6.79623 2.71849 4.97056 2.71849 3.84453 3.84453Z"
                fill="#fde047"
              />
              <path
                d="M10.5679 6.49012C10.556 6.50386 10.5435 6.51728 10.5304 6.53033L6.53044 10.5303C6.51735 10.5434 6.5039 10.5559 6.49011 10.5678L16.0777 20.1555C17.2038 21.2815 19.0294 21.2815 20.1555 20.1555C21.2815 19.0294 21.2815 17.2038 20.1555 16.0777L10.5679 6.49012Z"
                fill="#fde047"
              />
              <path
                d="M16.1 2.30719C16.261 1.8976 16.8385 1.8976 16.9994 2.30719L17.4298 3.40247C17.479 3.52752 17.5776 3.62651 17.7022 3.67583L18.7934 4.1078C19.2015 4.26934 19.2015 4.849 18.7934 5.01054L17.7022 5.44252C17.5776 5.49184 17.479 5.59082 17.4298 5.71587L16.9995 6.81115C16.8385 7.22074 16.261 7.22074 16.1 6.81116L15.6697 5.71587C15.6205 5.59082 15.5219 5.49184 15.3973 5.44252L14.3061 5.01054C13.898 4.849 13.898 4.26934 14.3061 4.1078L15.3973 3.67583C15.5219 3.62651 15.6205 3.52752 15.6697 3.40247L16.1 2.30719Z"
                fill="#fde047"
              />
              <path
                d="M19.9672 9.12945C20.1281 8.71987 20.7057 8.71987 20.8666 9.12945L21.0235 9.5288C21.0727 9.65385 21.1713 9.75284 21.2959 9.80215L21.6937 9.95965C22.1018 10.1212 22.1018 10.7009 21.6937 10.8624L21.2959 11.0199C21.1713 11.0692 21.0727 11.1682 21.0235 11.2932L20.8666 11.6926C20.7057 12.1022 20.1281 12.1022 19.9672 11.6926L19.8103 11.2932C19.7611 11.1682 19.6625 11.0692 19.5379 11.0199L19.14 10.8624C18.732 10.7009 18.732 10.1212 19.14 9.95965L19.5379 9.80215C19.6625 9.75284 19.7611 9.65385 19.8103 9.5288L19.9672 9.12945Z"
                fill="#fde047"
              />
              <path
                d="M5.1332 15.3072C5.29414 14.8976 5.87167 14.8976 6.03261 15.3072L6.18953 15.7065C6.23867 15.8316 6.33729 15.9306 6.46188 15.9799L6.85975 16.1374C7.26783 16.2989 7.26783 16.8786 6.85975 17.0401L6.46188 17.1976C6.33729 17.2469 6.23867 17.3459 6.18953 17.471L6.03261 17.8703C5.87167 18.2799 5.29414 18.2799 5.1332 17.8703L4.97628 17.471C4.92714 17.3459 4.82852 17.2469 4.70393 17.1976L4.30606 17.0401C3.89798 16.8786 3.89798 16.2989 4.30606 16.1374L4.70393 15.9799C4.82852 15.9306 4.92714 15.8316 4.97628 15.7065L5.1332 15.3072Z"
                fill="#fde047"
              />
            </svg>
            Magic Eraser!
          </h1>
          <p class="mt-8 text-pretty text-lg font-medium text-gray-500 sm:text-xl/8">
            基于 LaMA 模型实现的智能图像消除应用，可在 RISC-V64 平台上流畅运行。本页面由 Vue 与
            tailwind css 强力构建。
          </p>
          <div class="flex mt-10 items-center justify-center gap-x-6">
            <label
              for="uploadImage"
              class="flex bg-indigo-600 hover:bg-indigo-400 shadow-lg shadow-indigo-500/50 text-white text-base px-5 py-3 outline-none rounded-md w-max cursor-pointer font-[sans-serif]"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                class="w-6 mr-2 fill-white inline"
                viewBox="0 0 32 32"
              >
                <path
                  d="M23.75 11.044a7.99 7.99 0 0 0-15.5-.009A8 8 0 0 0 9 27h3a1 1 0 0 0 0-2H9a6 6 0 0 1-.035-12 1.038 1.038 0 0 0 1.1-.854 5.991 5.991 0 0 1 11.862 0A1.08 1.08 0 0 0 23 13a6 6 0 0 1 0 12h-3a1 1 0 0 0 0 2h3a8 8 0 0 0 .75-15.956z"
                  data-original="#000000"
                />
                <path
                  d="M20.293 19.707a1 1 0 0 0 1.414-1.414l-5-5a1 1 0 0 0-1.414 0l-5 5a1 1 0 0 0 1.414 1.414L15 16.414V29a1 1 0 0 0 2 0V16.414z"
                  data-original="#000000"
                />
              </svg>
              Get Started
              <input
                type="file"
                id="uploadImage"
                class="hidden"
                @change="handleImageUpload"
                accept="image/*"
              />
            </label>
            <a href="#" class="text-sm/6 font-semibold text-indigo-600 hover:text-indigo-400">
              More Info
              <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
      </div>
      <div v-if="this.isEdit" class="h-full w-full flex flex-col">
        <!-- Toolbar -->
        <div class="flex items-center justify-between px-4 py-8 drop-shadow-lg z-10">
          <div class="flex items-center space-x-4">
            <button
              @click="undo"
              :disabled="!canUndo"
              class="px-3 py-2 bg-gray-300 text-gray-700 rounded-md disabled:opacity-50"
              :class="canUndo ? hover : bg - gray - 400 ? hover : none"
            >
              Undo
            </button>
            <button
              @click="redo"
              :disabled="!canRedo"
              class="px-3 py-2 bg-gray-300 text-gray-700 rounded-md disabled:opacity-50"
              :class="canRedo ? hover : bg - gray - 400 ? hover : none"
            >
              Redo
            </button>
            <button
              class="px-3 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600 disabled:opacity-50"
              @click="genMaskAndCommit"
            >
              Commit
            </button>
            <button
              @click="saveImg"
              class="px-3 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
            >
              Save
            </button>
            <label
              class="px-3 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 cursor-pointer"
            >
              Upload
              <input
                type="file"
                id="uploadImage"
                class="hidden"
                @change="handleImageUpload"
                accept="image/*"
              />
            </label>
          </div>

          <div class="flex items-center space-x-4 px-5">
            <button
              title="Brush"
              class="hover:opacity-80 text-3xl flex items-center space-x-2"
              @click="editMask(0)"
              :class="toolMode == 0 ? 'opacity-100' : 'opacity-50'"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                width="32"
                height="32"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M11.4001 18.1612L11.4001 18.1612L18.796 10.7653C17.7894 10.3464 16.5972 9.6582 15.4697 8.53068C14.342 7.40298 13.6537 6.21058 13.2348 5.2039L5.83882 12.5999L5.83879 12.5999C5.26166 13.1771 4.97307 13.4657 4.7249 13.7838C4.43213 14.1592 4.18114 14.5653 3.97634 14.995C3.80273 15.3593 3.67368 15.7465 3.41556 16.5208L2.05445 20.6042C1.92743 20.9852 2.0266 21.4053 2.31063 21.6894C2.59466 21.9734 3.01478 22.0726 3.39584 21.9456L7.47918 20.5844C8.25351 20.3263 8.6407 20.1973 9.00498 20.0237C9.43469 19.8189 9.84082 19.5679 10.2162 19.2751C10.5343 19.0269 10.823 18.7383 11.4001 18.1612Z"
                  fill="#ffcd42"
                />
                <path
                  d="M20.8482 8.71306C22.3839 7.17735 22.3839 4.68748 20.8482 3.15178C19.3125 1.61607 16.8226 1.61607 15.2869 3.15178L14.3999 4.03882C14.4121 4.0755 14.4246 4.11268 14.4377 4.15035C14.7628 5.0875 15.3763 6.31601 16.5303 7.47002C17.6843 8.62403 18.9128 9.23749 19.85 9.56262C19.8875 9.57563 19.9245 9.58817 19.961 9.60026L20.8482 8.71306Z"
                  fill="#ffcd42"
                />
              </svg>
            </button>
            <button
              title="Eraser"
              class="hover:opacity-80 text-3xl flex items-center space-x-2"
              @click="editMask(1)"
              :class="toolMode == 1 ? 'opacity-100' : 'opacity-50'"
            >
              <svg
                viewBox="0 0 24 24"
                width="32"
                height="32"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M11.4096 5.50506C13.0796 3.83502 13.9146 3 14.9522 3C15.9899 3 16.8249 3.83502 18.4949 5.50506C20.165 7.1751 21 8.01013 21 9.04776C21 10.0854 20.165 10.9204 18.4949 12.5904L14.3017 16.7837L7.21634 9.69828L11.4096 5.50506Z"
                  fill="#ffcd42"
                />
                <path
                  d="M6.1557 10.759L13.2411 17.8443L12.5904 18.4949C12.2127 18.8727 11.8777 19.2077 11.5734 19.5H21C21.4142 19.5 21.75 19.8358 21.75 20.25C21.75 20.6642 21.4142 21 21 21H9C7.98423 20.9747 7.1494 20.1393 5.50506 18.4949C3.83502 16.8249 3 15.9899 3 14.9522C3 13.9146 3.83502 13.0796 5.50506 11.4096L6.1557 10.759Z"
                  fill="#ffcd42"
                />
              </svg>
            </button>
            <button
              title="Lasso"
              class="hover:opacity-80 text-3xl flex items-center space-x-2"
              @click="editMask(2)"
              :class="toolMode == 2 ? 'opacity-100' : 'opacity-50'"
            >
              <svg
                viewBox="0 0 24 24"
                width="32"
                height="32"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6.65376 1.63257C6.45076 1.27151 5.99349 1.14338 5.63244 1.34638C5.27138 1.54938 5.14324 2.00664 5.34624 2.3677L11.1396 12.6718L8.43226 17.4871C7.85064 16.1697 6.5327 15.2501 5 15.2501C2.92893 15.2501 1.25 16.9291 1.25 19.0001C1.25 21.0712 2.92893 22.7501 5 22.7501C6.42286 22.7501 7.66063 21.9577 8.29606 20.7901L12 14.2022L15.7039 20.79C16.3393 21.9577 17.5771 22.7501 19 22.7501C21.0711 22.7501 22.75 21.0712 22.75 19.0001C22.75 16.9291 21.0711 15.2501 19 15.2501C17.4673 15.2501 16.1494 16.1696 15.5677 17.4871L12.8604 12.6718L18.6538 2.3677C18.8568 2.00664 18.7286 1.54938 18.3676 1.34638C18.0065 1.14338 17.5492 1.27151 17.3462 1.63257L12 11.1415L6.65376 1.63257Z"
                  fill="#ffcd42"
                />
              </svg>
            </button>
            <button
              @click="clearCanvasByID('maskCanvas')"
              class="px-3 py-2 bg-red-500 text-white rounded-md hover:bg-red-600"
              title="Clear Mask"
            >
              Clear Mask
            </button>
          </div>

          <div class="flex items-center space-x-2">
            <label for="brush-size" class="text-gray-700">Brush Size:</label>
            <input
              type="range"
              id="brush-size"
              min="1"
              max="100"
              v-model="this.brushSize"
              class="cursor-pointer"
              @input="drawCircle()"
              @mouseup="clearCanvasByID('brushSizeCanvas')"
            />
          </div>

          <div
            id="alert"
            class="hidden fixed left-0 right-0 rounded-md opacity-90 z-50 bg-yellow-500 text-white text-center p-3"
          >
            <span>Processing... Please wait!</span>
          </div>
        </div>

        <!-- Canvas Container -->
        <div id="canvasContainer" class="flex-grow overflow-hidden relative justify-center">
          <div
            v-if="this.imgHeight != 0"
            class="absolute top-5 left-5 bg-black rounded-md px-3 py-2 opacity-80 hover:opacity-50 z-20"
          >
            <p class="text-white">{{ imgWidth }} x {{ imgHeight }}</p>
          </div>
          <div
            v-if="this.imgHeight != 0"
            class="absolute top-5 right-5 bg-yellow-500 rounded-md px-3 py-2 opacity-80 z-20 space-x-5 flex"
          >
            <button
              title="zoomIn"
              @click="zoomIn"
              class="hover:bg-yellow-700/50 rounded-md flex items-center space-x-2"
            >
              <svg
                viewBox="0 0 24 24"
                width="24"
                height="24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                stroke="#6b4d99"
                stroke-width="0.72"
              >
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M4 11C4 7.13401 7.13401 4 11 4C14.866 4 18 7.13401 18 11C18 14.866 14.866 18 11 18C7.13401 18 4 14.866 4 11ZM11 2C6.02944 2 2 6.02944 2 11C2 15.9706 6.02944 20 11 20C13.125 20 15.078 19.2635 16.6177 18.0319L20.2929 21.7071C20.6834 22.0976 21.3166 22.0976 21.7071 21.7071C22.0976 21.3166 22.0976 20.6834 21.7071 20.2929L18.0319 16.6177C19.2635 15.078 20 13.125 20 11C20 6.02944 15.9706 2 11 2Z"
                  fill="#6b4d99"
                />
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M10 14C10 14.5523 10.4477 15 11 15C11.5523 15 12 14.5523 12 14V12H14C14.5523 12 15 11.5523 15 11C15 10.4477 14.5523 10 14 10H12V8C12 7.44772 11.5523 7 11 7C10.4477 7 10 7.44772 10 8V10H8C7.44772 10 7 10.4477 7 11C7 11.5523 7.44772 12 8 12H10V14Z"
                  fill="#6b4d99"
                />
              </svg>
            </button>
            <button
              title="zoomOut"
              @click="zoomOut"
              class="hover:bg-yellow-700/50 rounded-md flex items-center space-x-2"
            >
              <svg
                viewBox="0 0 20 20"
                width="24"
                height="24"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                stroke="#6b4d99"
                stroke-width="0.6"
              >
                <path
                  fill="#6b4d99"
                  fill-rule="evenodd"
                  d="M9 4a5 5 0 100 10A5 5 0 009 4zM2 9a7 7 0 1112.6 4.2.999.999 0 01.107.093l3 3a1 1 0 01-1.414 1.414l-3-3a.999.999 0 01-.093-.107A7 7 0 012 9zm10.5 0a1 1 0 00-1-1h-5a1 1 0 100 2h5a1 1 0 001-1z"
                />
              </svg>
            </button>
            <button
              title="reset"
              @click="reset"
              class="hover:bg-yellow-700/50 rounded-md flex items-center space-x-2"
            >
              <svg
                fill="#6b4d99"
                viewBox="-192 -192 2304.00 2304.00"
                xmlns="http://www.w3.org/2000/svg"
                stroke="#6b4d99"
                stroke-width="57.6"
                width="24"
                height="24"
              >
                <path
                  d="M960 0v213.333c411.627 0 746.667 334.934 746.667 746.667S1371.627 1706.667 960 1706.667 213.333 1371.733 213.333 960c0-197.013 78.4-382.507 213.334-520.747v254.08H640V106.667H53.333V320h191.04C88.64 494.08 0 720.96 0 960c0 529.28 430.613 960 960 960s960-430.72 960-960S1489.387 0 960 0"
                  fill-rule="evenodd"
                />
              </svg>
            </button>
          </div>
          <div class="absolute inset-0 flex items-center justify-center z-0">
            <canvas id="brushSizeCanvas" width="1" height="1"></canvas>
          </div>
          <div class="absolute inset-0 flex items-center justify-center z-10">
            <canvas id="maskCanvas" width="1" height="1"></canvas>
          </div>
          <div class="absolute inset-0 flex items-center justify-center -z-10">
            <canvas id="fabricCanvas" width="1" height="1"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
class HistoryStack {
  constructor() {
    this.history = []
    this.redoStack = []
    this.currentIndex = -1
  }

  addState(state) {
    this.history = this.history.slice(0, this.currentIndex + 1)
    this.history.push(state)
    this.currentIndex++
    this.redoStack = []
  }

  undo() {
    if (this.canUndo()) {
      const state = this.history[this.currentIndex]
      this.redoStack.push(state)
      this.currentIndex--
      return this.getCurrentState()
    }
    return null
  }

  redo() {
    if (this.canRedo()) {
      this.currentIndex++
      const state = this.redoStack.pop()
      this.history.push(state)
      return state
    }
    return null
  }

  getCurrentState() {
    if (this.currentIndex > -1) {
      return this.history[this.currentIndex]
    }
    return null
  }

  canUndo() {
    return this.currentIndex > 0
  }

  canRedo() {
    return this.redoStack.length > 0
  }

  clear() {
    this.history = []
    this.redoStack = []
    this.currentIndex = -1
  }
}


export default {
  data() {
    return {
      // States
      isEdit: false,
      toolMode: -1, // 0-Brush, 1-Eraser, 2-Lasso
      // Image Attributes
      ImgURL: null,
      imgType: null,
      imgHeight: 0,
      imgWidth: 0,
      // History stack
      historyStack: new HistoryStack(),
      canUndo: false,
      canRedo: false,
      // Others
      initZoomRate: 1,
      zoomRate: 1,
      brushSize: 10,
    }
  },
  methods: {
    handleImageUpload(event) {
      const file = event.target.files[0]
      if (file) {
        const reader = new FileReader()
        reader.onload = (e) => {
          // Navigate to WorkPlace with the uploaded image data URL
          this.historyStack.clear()
          const imgURL = e.target.result
          this.storeHistory(imgURL)
          this.loadImgToCanvas(imgURL)
        }
        reader.readAsDataURL(file)
        this.isEdit = true
        this.imgType = file.type
      }
    },
    clearCanvas() {
      // Clean History
      this.toolMode = -1
      this.clearCanvasByID('fabricCanvas')
      this.clearCanvasByID('maskCanvas')
      this.clearCanvasByID('brushSizeCanvas')
    },
    loadImgToCanvas(img) {
      this.clearCanvas()

      const canvas = document.getElementById('fabricCanvas')
      const maskCanvas = document.getElementById('maskCanvas')
      const brushSizeCanvas = document.getElementById('brushSizeCanvas')
      const canvasContainer = document.getElementById('canvasContainer')

      this.ImgURL = img

      var image = new Image()
      image.src = img
      console.log('Upload Img: ' + img)
      image.onload = () => {
        const ctx = canvas.getContext('2d')
        console.log('ctx: ' + ctx)
        console.log('Width: ' + image.width)
        console.log('Height: ' + image.height)

        this.imgWidth = image.width
        this.imgHeight = image.height

        const canvasWidth = canvasContainer.offsetWidth
        const canvasHeight = canvasContainer.offsetHeight
        console.log('Width: ' + canvasWidth)
        console.log('Height: ' + canvasHeight)

        const widthScaleRate = (canvasWidth - 10) / image.width
        const heightScaleRate = (canvasHeight - 10) / image.height
        const scaleRate = heightScaleRate > widthScaleRate ? widthScaleRate : heightScaleRate

        ctx.scale(scaleRate, scaleRate)
        this.initZoomRate = scaleRate
        this.zoomRate = scaleRate
        canvas.width = image.width * scaleRate
        canvas.height = image.height * scaleRate
        maskCanvas.width = canvas.width
        maskCanvas.height = canvas.height
        brushSizeCanvas.width = canvas.width
        brushSizeCanvas.height = canvas.height

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
      }

      this.enableDrag()
    },
    storeHistory(imgURL) {
      this.historyStack.addState(imgURL)
      this.updateHistoryPermi()
    },
    undo() {
      const prev = this.historyStack.undo()
      if (prev) {
        this.updateHistoryPermi()
        this.loadImgToCanvas(prev)
      }
    },
    redo() {
      const next = this.historyStack.redo()
      if (next) {
        this.updateHistoryPermi()
        this.loadImgToCanvas(next)
      }
    },
    updateHistoryPermi() {
      this.canUndo = this.historyStack.canUndo()
      this.canRedo = this.historyStack.canRedo()
    },
    editMask(toolMode) {
      const canvas = document.getElementById('maskCanvas')
      const ctx = canvas.getContext('2d')
      const brushSizeCanvas = document.getElementById('brushSizeCanvas')
      const brushSizeCtx = brushSizeCanvas.getContext('2d')

      var isDrawing = false
      var lastX = 0
      var lastY = 0
      var lassoPoints = []

      this.toolMode = toolMode
      brushSizeCtx.clearRect(0, 0, canvas.width, canvas.height)

      canvas.addEventListener('mouseout', () => {
        brushSizeCtx.clearRect(0, 0, canvas.width, canvas.height)
      })

      canvas.addEventListener('mousedown', (e) => {
        if (e.button === 0) {
          isDrawing = true
            ;[lastX, lastY] = [e.offsetX, e.offsetY]
          if (this.toolMode == 2) {
            lassoPoints = []
            lassoPoints.push([lastX, lastY])
          }
        }
      })

      canvas.addEventListener('mousemove', (e) => {
        if (this.toolMode == 0 || this.toolMode == 1) {
          // Show brushSize

          brushSizeCtx.clearRect(0, 0, canvas.width, canvas.height)
          brushSizeCtx.beginPath()
          brushSizeCtx.arc(e.offsetX, e.offsetY, this.brushSize, 0, Math.PI * 2, false)
          brushSizeCtx.fillStyle = 'rgba(250, 204, 21, 0.5)'
          brushSizeCtx.fill()
          brushSizeCtx.closePath()
        }

        if (isDrawing) {
          if (this.toolMode == 0 || this.toolMode == 2) {
            drawMask(e.offsetX, e.offsetY, this.brushSize, this.toolMode)
          } else if (this.toolMode == 1) {
            eraseMask(e.offsetX, e.offsetY, this.brushSize)
          }
        }
      })

      canvas.addEventListener('mouseup', () => {
        if (this.toolMode == 2) {
          ctx.beginPath()
          ctx.moveTo(lassoPoints[0][0], lassoPoints[0][1])
          lassoPoints.forEach((point) => {
            ctx.lineTo(point[0], point[1])
          })
          ctx.closePath()
          ctx.fillStyle = 'rgb(250 204 21)'
          ctx.fill()

          lassoPoints = []
        }
        isDrawing = false
      })

      ctx.save()

      function drawMask(x, y, size, mode) {
        if (mode == 2) {
          lassoPoints.push([x, y])
          ctx.lineWidth = 1
        } else {
          ctx.lineWidth = size * 2
        }

        ctx.beginPath()
        ctx.moveTo(lastX, lastY)
        ctx.lineTo(x, y)
        ctx.lineCap = 'round'
        ctx.strokeStyle = 'rgb(250 204 21)'
        ctx.stroke()
          ;[lastX, lastY] = [x, y]
      }

      function eraseMask(x, y, size) {
        ctx.globalCompositeOperation = 'destination-out'
        ctx.beginPath()
        ctx.moveTo(lastX, lastY)
        ctx.lineTo(x, y)
        ctx.strokeStyle = 'rgba(0, 0, 0, 1)'
        ctx.lineWidth = size * 2
        ctx.lineCap = 'round'
        ctx.stroke()
        ctx.globalCompositeOperation = 'source-over'
          ;[lastX, lastY] = [x, y]
      }
    },
    drawCircle() {
      const canvas = document.getElementById('brushSizeCanvas')
      const ctx = canvas.getContext('2d')
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.beginPath()
      ctx.arc(centerX, centerY, this.brushSize, 0, Math.PI * 2, false)
      ctx.fillStyle = 'rgba(250, 204, 21, 0.7)'
      ctx.fill()
      ctx.closePath()
    },
    clearCanvasByID(canvasID) {
      const canvas = document.getElementById(canvasID)
      const ctx = canvas.getContext('2d')
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    },
    genMaskAndCommit() {
      const imgCanvas = document.getElementById('maskCanvas');
      const ctx = imgCanvas.getContext('2d');
      const maskImg = ctx.getImageData(0, 0, imgCanvas.clientWidth, imgCanvas.clientHeight);
      const maskData = maskImg.data;
      const newMaskImg = new ImageData(this.imgWidth, this.imgHeight);
      const newMaskData = newMaskImg.data;
      const widthRatio = imgCanvas.clientWidth / this.imgWidth;
      const heightRatio = imgCanvas.clientHeight / this.imgHeight;
      for (let i = 0; i <this.imgHeight; i++) {
        for (let j = 0; j < this.imgWidth; j++) {
          const originalX = Math.floor(j * widthRatio);
          const originalY = Math.floor(i * heightRatio);
          const originalIndex = (originalY * imgCanvas.clientWidth + originalX) * 4;
          const newIndex = (i * this.imgWidth + j) * 4;
          if (maskData[originalIndex + 3] < 128) {
            newMaskData[newIndex] = 0; // Red
            newMaskData[newIndex + 1] = 0; // Green
            newMaskData[newIndex + 2] = 0; //Blue 
            newMaskData[newIndex + 3] = 255; // Alpha
          } 
          else {
            newMaskData[newIndex] = 255; // Red
            newMaskData[newIndex + 1] = 255; //Green 
            newMaskData[newIndex + 2] = 255; // Blue
            newMaskData[newIndex + 3] = 255; // Alpha
          } 
        } 
      }
      const exportMask = document.createElement('canvas');
      exportMask.width = this.imgWidth; exportMask.height = this.imgHeight; const
        exportCtx = exportMask.getContext('2d'); exportCtx.putImageData(newMaskImg, 0,
          0); const img = new Image(); img.src = exportMask.toDataURL();

      document.getElementById("alert").classList.remove('hidden')
      this.toolMode = -1

      img.onload = () => {
        fetch('http://127.0.0.1:7777/predict', {
          method: 'POST',
          body: JSON.stringify({ 'image': this.ImgURL, 'mask': exportMask.toDataURL() }),
          headers: {
            'Content-Type': 'application/json',
          }
        })
          .then(response => response.json())
          .then(data => {
            if (data.result_image) {
              document.getElementById("alert").classList.add('hidden')
              const resultURL = data.result_image
              this.storeHistory(resultURL)
              this.loadImgToCanvas(resultURL)
            }
          })
          .catch(error => {
            console.error('Error in processing Image: ', error.message)
          })
      }
    },
    scaleImage(src, width, height) {
      const img = new Image()
      img.src = src
      img.onload = () => {
        const tmpCanvas = document.createElement('canvas')
        tmpCanvas.width = width
        tmpCanvas.height = height
        const ctx = tmpCanvas.getContext('2d')
        ctx.drawImage(img, 0, 0, width, height)
        return tmpCanvas.toDataURL('image/png')
      }
    },
    saveImg() {
      const canvas = document.getElementById('fabricCanvas')
      const dataURL = canvas.toDataURL(this.imgType)

      const timestamp = Date.now()

      const link = document.createElement('a')
      link.href = dataURL
      link.download = String(timestamp) + '.' + this.imgType.split('/')[1]
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    },
    zoomIn() {
      this.zoomRate *= 1.1
      this.updateCanvas()
    },
    zoomOut() {
      console.log('zoomOut')
      this.zoomRate /= 1.1
      this.updateCanvas()
    },
    enableDrag() {
      this.enableCanvasDrag('fabricCanvas')
      this.enableCanvasDrag('maskCanvas')
      this.enableCanvasDrag('brushSizeCanvas')
    },
    reset() {
      this.zoomRate = this.initZoomRate
      this.updateCanvas()

      const canvas = document.getElementById('fabricCanvas')
      const maskCanvas = document.getElementById('maskCanvas')
      const brushSizeCanvas = document.getElementById('brushSizeCanvas')
      canvas.style.transform = 'translate(0px, 0px)'
      maskCanvas.style.transform = 'translate(0px, 0px)'
      brushSizeCanvas.style.transform = 'translate(0px, 0px)'
    },
    // Sizing Tools
    updateCanvas() {
      const canvas = document.getElementById('fabricCanvas')
      const maskCanvas = document.getElementById('maskCanvas')
      const brushSizeCanvas = document.getElementById('brushSizeCanvas')
      const ctx = canvas.getContext('2d')
      const maskCtx = maskCanvas.getContext('2d')

      const image = new Image()
      image.src = this.ImgURL
      image.onload = () => {
        const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)

        const newWidth = this.imgWidth * this.zoomRate
        const newHeight = this.imgHeight * this.zoomRate
        canvas.width = newWidth
        canvas.height = newHeight
        maskCanvas.width = newWidth
        maskCanvas.height = newHeight
        brushSizeCanvas.width = newWidth
        brushSizeCanvas.height = newHeight

        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(image, 0, 0, newWidth, newHeight)

        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
        maskCtx.putImageData(this.scaleMask(maskImageData, newWidth / maskImageData.width), 0, 0)
      }
    },
    scaleMask(maskImageData, scale) {
      const tmpCanvas = document.createElement('canvas')
      tmpCanvas.width = maskImageData.width
      tmpCanvas.height = maskImageData.height
      const tmpCtx = tmpCanvas.getContext('2d')
      tmpCtx.putImageData(maskImageData, 0, 0)

      const scaledCanvas = document.createElement('canvas')
      scaledCanvas.width = maskImageData.width * scale
      scaledCanvas.height = maskImageData.height * scale
      const scaledCtx = scaledCanvas.getContext('2d')
      scaledCtx.imageSmoothingEnabled = false
      scaledCtx.drawImage(tmpCanvas, 0, 0, scaledCanvas.width, scaledCanvas.height)

      return scaledCtx.getImageData(0, 0, scaledCanvas.width, scaledCanvas.height)
    },
    enableCanvasDrag(canvasId) {
      const canvas = document.getElementById(canvasId)
      const canvasContainer = document.getElementById('canvasContainer')
      let isDragging = false
      let lastMouseX = 0
      let lastMouseY = 0
      let offsetX = 0
      let offsetY = 0

      canvasContainer.oncontextmenu = (e) => e.preventDefault()

      canvasContainer.addEventListener('mousedown', (e) => {
        if (e.button === 2) {
          isDragging = true
          lastMouseX = e.clientX
          lastMouseY = e.clientY
        }
      })

      canvasContainer.addEventListener('mousemove', (e) => {
        if (isDragging) {
          const deltaX = e.clientX - lastMouseX
          const deltaY = e.clientY - lastMouseY
          offsetX += deltaX
          offsetY += deltaY
          lastMouseX = e.clientX
          lastMouseY = e.clientY

          canvas.style.transform = `translate(${offsetX}px, ${offsetY}px)`
        }
      })

      canvasContainer.addEventListener('mouseup', (e) => {
        if (e.button === 2) {
          isDragging = false
        }
      })

      canvasContainer.addEventListener('mouseleave', () => {
        isDragging = false
      })
    },
  },
}
</script>
