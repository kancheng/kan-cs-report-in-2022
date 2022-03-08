<template>
  <img alt="Vue logo" src="./assets/logo.png">
  <HelloWorld msg="Welcome to Your Vue.js App"/>
  <HelloWorldTest msg="OwO /// Testing Kan !!"/>
</template>

<script>
import HelloWorld from './components/HelloWorld.vue'
import HelloWorldTest from './components/HelloWorldTest.vue'

export default {
  name: 'App',
  components: {
    HelloWorld, HelloWorldTest
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
