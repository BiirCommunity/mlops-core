FROM node:22-alpine AS build
WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .
ARG VITE_API_BASE_URL=/api
ARG VITE_MLFLOW_UI_URL=http://localhost:30500
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
ENV VITE_MLFLOW_UI_URL=$VITE_MLFLOW_UI_URL
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
